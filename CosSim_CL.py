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
from tqdm import tqdm
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# export HF_HOME=/dev/cache/huggingface/
# export HF_DATASETS_OFFLINE=1


def get_pruned_layer_with_cosine(model, trainloader, num_to_prune, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    max_start = num_layers - num_to_prune
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    cosine_sim = [torch.zeros(1).to(device) for _ in range(max_start)]

    def hook(module, input, output, layer_name):
        # norm_input = F.normalize(input[0], p=2, dim=-1)
        act[layer_name] = input[0] #+ d[layer_name]
    handles = []

    for l, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(lambda module, input, output, layer_name=l: hook(module, input, output, layer_name))
        handles.append(handle)

    num_samples = num_sample = 128
    select_loop = tqdm(enumerate(trainloader, start=1), desc="Selecting", total=num_samples)
    for num, batch in select_loop:
        batch = batch[0].to(device)
        try:
            with torch.no_grad():
                output = model(batch)
        except IndexError:
            pass
        num_sample -= 1

        for i in range(1, max_start):
            cosine_sim[i] += torch.cosine_similarity(act[i], act[i+num_to_prune]).mean()
            # x, y = act[i], act[i+num_to_prune]
            # norm_x = torch.norm(x, dim=-1, p=2, keepdim=True)
            # norm_y = torch.norm(y, dim=-1, p=2, keepdim=True)
            # cosine = torch.sum(x * y) / (norm_x * norm_y)
            # cosine_sim[i] += cosine.mean()

        if not num_sample:
            break

    for handle in handles:
        handle.remove()

    cosine_sim = [i.item()/num_samples for i in cosine_sim]
    max_sim = max(cosine_sim)
    start_l = cosine_sim.index(max_sim)
    end_l = start_l + num_to_prune

    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, max_sim


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

    cal_loader = get_wikitext2(tokenizer, train_size=128, val_size=64, seed=0, seqlen=2048, test_only=False)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir, dist_rank=args.model.split("/")[-1])

    layer, start_l, end_l, max_value = get_pruned_layer_with_cosine(model, cal_loader, args.num_to_prune, dev)

    scale = get_scale_params(model, cal_loader, start_l, end_l, dev)
    print(scale)
    model.model.layers = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i < start_l or i >= end_l])
    fuse_scale(model, scale, start_l)

    remove_layer = [l for l in range(start_l, end_l)]
    print(f'remove layer idx: {remove_layer}')
    logger.info(f'remove layer idx: {remove_layer}')

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