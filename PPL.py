import os
import sys
import random
import numpy as np
import torch
from eval_utils import evaluate, create_logger
from magcomp import fuse_scale, get_scale_params
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
from data_utils.calibration_dataset import get_wikitext2
from tqdm import tqdm
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)
os.environ["TOKENIZERS_PARALLELISM"] = "true"



def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset("/path_to_datasets/wikitext", 'wikitext-2-raw-v1',
                             split='train')  # load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc


@torch.no_grad()
def get_loss(model, testenc, bs, seqlen, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    # List to store negative log likelihoods
    losses = []
    # print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * (j - i)

        # Append to list of negative log likelihoods
        losses.append(loss)

    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum()

    return loss_sum.item()

def turn_off(model, j):

    def hook(module, input, output):
        return input

    for l, block in enumerate(model.model.layers):
        if l==j:
            handle = block.register_forward_hook(hook)
    return handle


def sleb(model, dataloader):
    early_barrier = 1
    latter_barrier = 1
    use_cache = model.config.use_cache
    model.config.use_cache = False
    num_blocks = len(model.model.layers)
    alive_list = [i for i in range(num_blocks)]
    removal_list = []
    loss_list = [1e99 for _ in range(early_barrier)]

    model.eval()

    min_loss = 1e99
    min_loss_idx = -1

    search_bound = num_blocks

    for j in range(early_barrier, search_bound - latter_barrier):

        # kill j-th alive block
        handle = turn_off(model, j)

        loss = get_loss(model, dataloader, bs=1, seqlen=2048, device=torch.device("cuda:0"))
        torch.cuda.empty_cache()
        loss_list.append(loss)

        if loss < min_loss:
            min_loss = loss
            min_loss_idx = j

        print(
            f"[Block {j} (Original block {alive_list[j]}) removed] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {alive_list[min_loss_idx]}"
        )
        # unkill j-th alive block

        handle.remove()


    print(loss_list)

    indices = sorted(range(len(loss_list)), key=lambda i: loss_list[i], reverse=False)
    print(indices)

    del alive_list[min_loss_idx]

    return indices

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--num_to_prune", type=int, default=0, help="number of layers to prune")
    parser.add_argument("--log_dir", default=None, type=str, help="direction of logging file")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_mmlu", action="store_true", help="evaluate MMLU")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--max_memory", type=str, default="70GiB", help="The maximum memory of each GPU")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving checkpoints")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model) #, use_fast=False, legacy=False
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model = model.to(dev).half()
    before_pruning_parameters = sum(p.numel() for p in model.parameters())

    num_layers = len(model.model.layers)
    layers_state = [False for _ in range(num_layers)]

    dataloader = get_wikitext2_trainenc(seed=0, nsamples=128, seqlen=2048, model=model, tokenizer=tokenizer, batch_size=1)

    cal_loader = get_wikitext2(tokenizer, train_size = 128, val_size = 64, seed = 0, seqlen = 2048, test_only = False)

    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir, dist_rank=args.model.split("/")[-1])

    remove_layer = []

    for l in range(args.num_to_prune):

        indices = sleb(model, dataloader)
        start_l, end_l = indices[0], indices[0] + 1

        scale = get_scale_params(model, cal_loader, start_l, end_l, dev)
        print(scale)
        model.model.layers = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i < start_l or i >= end_l])
        fuse_scale(model, scale, start_l)

        remaining_layers = [i for i, state in enumerate(layers_state) if not state]
        ori_idx = remaining_layers[start_l]
        layers_state[ori_idx] = True
        remove_layer.append(ori_idx)

        if l == 0:
            print(f'initialized layer importance: {indices}')
            logger.info(f'initialized layer importance: {indices}')
        print(f'remove layer idx of round {l}: {remove_layer}')
        logger.info(f'remove layer idx of round {l}: {remove_layer}')

    after_pruning_parameters = sum(p.numel() for p in model.parameters())

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info("start saving model")
        tokenizer.save_pretrained(args.save_dir)
        model.save_pretrained(args.save_dir)
        config = AutoConfig.from_pretrained(args.save_dir, trust_remote_code=True)
        config.num_hidden_layers = config.num_hidden_layers - args.num_to_prune
        config._name_or_path = args.save_dir
        config.save_pretrained(args.save_dir)
        logger.info("save model success")


    evaluate(model, tokenizer, logger, args.log_dir, eval_ppl=args.eval_ppl, eval_mmlu=args.eval_mmlu,
             eval_tasks=args.eval_tasks, eval_batch_size=1, max_memory='30GiB')

    logger.info(
        "#PruneLayer: {} #Param before: {}, #Param after: {}, PruneRatio = {:.4f}%".format(args.num_to_prune,
                                                                                           before_pruning_parameters,
                                                                                           after_pruning_parameters,
                                                                                           100 - 100.0 * after_pruning_parameters / before_pruning_parameters))
    print(
        "#PruneLayer: {} #Param before: {}, #Param after: {}, PruneRatio = {:.4f}%".format(args.num_to_prune,
                                                                                           before_pruning_parameters,
                                                                                           after_pruning_parameters,
                                                                                           100 - 100.0 * after_pruning_parameters / before_pruning_parameters))

if __name__ == "__main__":
    print(sys.argv)
    main()

