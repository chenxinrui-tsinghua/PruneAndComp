from data_utils.taylor_dataset import get_examples
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
# dev = torch.device("cpu")
print(dev)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def taylor_plus(model,
            batch_size,
            example_prompts,
            norm_power=1, # help="1 or 2 for l-p norm"
            weight_reduction="sum", #help="sum, mean, max, prod"
            block_reduction="sum", #help="sum, mean, max, prod"
            first_barrier=4,
            last_barrier=2):

    model = model.to("cpu").float()

    print("Do forward to collect gradient information")
    salience_dict = {}
    for i in tqdm(range(0, example_prompts.size(0), batch_size)):
        example_prompts_tmp = example_prompts[i : i + batch_size].to("cpu")
        loss = model(example_prompts_tmp, labels=example_prompts_tmp).loss
        loss.backward()
        for k, param in model.named_parameters():
            if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                salience = param * param.grad
                salience = salience.data.clone().float()

                if k not in salience_dict.keys():
                    salience_dict[k] = salience
                else:
                    salience_dict[k] += salience
        model.zero_grad()

    # Compute scores of weight matrices -> Collec them
    block_info = {}

    for k, param in model.named_parameters():
        if param.requires_grad and "weight" in k and "embed_tokens" not in k:
            block_idx = ".".join(k.split(".")[:3])  # 'model.layers.i'
            if "proj" in k or "lm_head" in k:  # output_dim x input_dim
                weight_imp = (
                    salience_dict[k].abs().pow(norm_power).sum(1)
                )  # [output_dim]
            elif "norm" in k:  # [output_dim]
                weight_imp = salience_dict[k].abs().pow(norm_power)

            if weight_reduction == "sum":
                weight_imp = weight_imp.sum(dim=0)
            elif weight_reduction == "mean":
                weight_imp = weight_imp.mean(dim=0)
            elif weight_reduction == "max":
                weight_imp = weight_imp.max(dim=0)[0]
            elif weight_reduction == "prod":
                weight_imp = torch.prod(weight_imp, dim=0)
            else:
                raise NotImplementedError

            weight_imp = weight_imp.item()

            print([k, weight_imp])
            if block_idx not in block_info.keys():
                block_info[block_idx] = [weight_imp]
            else:
                block_info[block_idx].append(weight_imp)

    # Compute block-level importance
    block_info_summary = {}

    for k, v in block_info.items():
        print(k, v)

        block_imp = torch.tensor(v)
        if block_reduction == "sum":
            block_imp = block_imp.sum(dim=0)
        elif block_reduction == "mean":
            block_imp = block_imp.mean(dim=0)
        elif block_reduction == "max":
            block_imp = block_imp.max(dim=0)[0]
        elif block_reduction == "prod":
            block_imp = torch.prod(block_imp, dim=0)
        else:
            raise NotImplementedError

        block_imp = block_imp.item()
        block_info_summary[k] = block_imp

    for k in ["model.norm.weight", "lm_head.weight"]:
        if k in block_info_summary:
            del block_info_summary[k]
    sorted_items = sorted(block_info_summary.items(), key=lambda x: x[1])
    block_order = []

    for rank, (key, value) in enumerate(sorted_items, start=1):
        print([rank, key, value, key.split(".")[-1]])
        block_order.append(int(key.split(".")[-1]))



    print(block_order)
    print(f"=== block order removed:")

    last_barrier = len(block_order) - last_barrier

    block_order = [l for l in block_order if first_barrier<=l<last_barrier]

    print(block_order)

    return block_order

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
    before_pruning_parameters = sum(p.numel() for p in model.parameters())
    model.config.use_cache = False

    num_layers = len(model.model.layers)
    layers_state = [False for _ in range(num_layers)]

    cal_loader = get_wikitext2(tokenizer, train_size=128, val_size=64, seed=0, seqlen=2048, test_only=False)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir, dist_rank=args.model.split("/")[-1])

    remove_layer = []


    # cahce_path = "./llama2-7b-prompt-wikitext2-1x2048-cache.pt"

    # if os.path.exists(cahce_path):
    #     example_prompts = torch.load(cahce_path)
    # else:
    #     example_prompts = get_examples(
    #         dataset="bookcorpus", #"c4", #"wikitext2", #
    #         tokenizer=tokenizer,
    #         n_samples=10,
    #         seq_len=2048,
    #         field_name="text",
    #         add_bos_to_every=False)
    #     torch.save(example_prompts, cahce_path)

    example_prompts = get_examples(
        dataset="c4", #"wikitext2", #"bookcorpus",  #
        tokenizer=tokenizer,
        n_samples=10,
        seq_len=2048,
        field_name="text",
        add_bos_to_every=False)

    for l in range(args.num_to_prune):

        print(example_prompts.shape)

        indices = taylor_plus(model,
            batch_size=1, #
            example_prompts=example_prompts,
            norm_power=1, # help="1 or 2 for l-p norm"
            weight_reduction="sum", #help="sum, mean, max, prod"
            block_reduction="sum", #help="sum, mean, max, prod"
            first_barrier=4,
            last_barrier=2)

        model = model.half().to(dev)  # .float()#

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

    # model.model.layers = nn.ModuleList(
    #     [layer for i, layer in enumerate(model.model.layers) if i==0])

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
