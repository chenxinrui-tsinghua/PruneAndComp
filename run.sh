# usage: bash run.sh
# Replace the correct paths for the models and datasets (run.sh, eval_utils.py,  files in data_utils folder) for evaluation

# One-shot Baseline of CosSim(BI)
CUDA_VISIBLE_DEVICES=0 python OneShotPrune.py \
--model /path_to_models/llama-3-8b \
--prune_idx 27 26 24 28 29 \
--log_dir /path/results/llama-3-8b/OneShotPrune-p5 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa" \
--save_dir /path/save/llama-3-8b/OneShotPrune-p5 # save_dir is optional

# CosSim(BI)
CUDA_VISIBLE_DEVICES=0 python CosSim_BI.py \
--model /path_to_models/llama-3-8b \
--num_to_prune 5 \
--log_dir /path/results/llama-3-8b/BI-p5 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa" \
--save_dir /path/save/llama-3-8b/BI-p5 # save_dir is optional

# CosSim(CL)
CUDA_VISIBLE_DEVICES=0 python CosSim_CL.py \
--model /path_to_models/llama-3-8b \
--num_to_prune 5 \
--log_dir /path/results/llama-3-8b/LLM-Streamline-p5 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"

# Taylor+
CUDA_VISIBLE_DEVICES=0 python Taylor+.py \
--model /path_to_models/llama-3-8b \
--num_to_prune 5 \
--log_dir /path/results/llama-3-8b/taylor-p5 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"

# Mag+
CUDA_VISIBLE_DEVICES=0 python mag+.py \
--model /path_to_models/llama-3-8b \
--num_to_prune 5 \
--log_dir /path/results/llama-3-8b/mag+p7 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"

# PPL
CUDA_VISIBLE_DEVICES=0 python PPL.py \
--model /path_to_models/llama-3-8b \
--num_to_prune 5 \
--log_dir /path/results/llama-3-8b/PPL-p5 \
--eval_ppl \
--eval_mmlu \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"