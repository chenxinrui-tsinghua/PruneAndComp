import os
import sys
import random
import numpy as np
import torch
from eval_utils import evaluate, create_logger
from magcomp import fuse_scale, get_scale_params
from data_utils.calibration_dataset import get_wikitext2
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# export HF_HOME=/dev/cache/huggingface/
# export HF_DATASETS_OFFLINE=1


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

def BI(model, trainloader):
    model = model.to(dev)
    num_layer = len(model.model.layers)
    BI = [0 for _ in range(num_layer)]

    def hook(module, input, output, layer_name):
        BI[layer_name] += block_influence(input[0], output[0], angular=False).sum().cpu().item()
        # print(BI)

    handles = []
    for l, block in enumerate(model.model.layers):
        handle = block.register_forward_hook(
            lambda module, input, output, layer_name=l: hook(module, input, output, layer_name))
        handles.append(handle)

    num_sample = 128
    for batch in trainloader:
        batch = batch[0].to(dev)
        try:
            with torch.no_grad():
                output = model(batch)
        except IndexError:
            pass

        num_sample -= 1
        if not num_sample:
            break

    indices = sorted(range(len(BI)), key=lambda i: BI[i], reverse=False)
    # print(indices)
    for handle in handles:
        handle.remove()

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

    cal_loader = get_wikitext2(tokenizer, train_size=256, val_size=64, seed=0, seqlen=2048, test_only=False)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir, dist_rank=args.model.split("/")[-1])

    remove_layer = []

    for l in range(args.num_to_prune):
        indices = BI(model, cal_loader)

        start_l, end_l = indices[0], indices[0] + 1

        scale = get_scale_params(model, cal_loader, start_l, end_l, dev, num_sample=256)
        print(scale)
        model.model.layers = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i < start_l or i >= end_l])
        fuse_scale(model, scale, start_l)

        remaining_layers = [i for i, state in enumerate(layers_state) if not state]
        ori_idx = remaining_layers[start_l]
        layers_state[ori_idx] = True
        remove_layer.append(ori_idx)

        if l==0:
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

    evaluate(model, tokenizer, logger, args.log_dir, eval_ppl=args.eval_ppl, eval_mmlu=args.eval_mmlu, eval_tasks=args.eval_tasks, eval_batch_size=1, max_memory='30GiB')

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