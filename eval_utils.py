import random
import torch.nn as nn
import torch
from datasets import load_dataset
from mmlu.mmlu_eval import run_mmlu_eval
import logging
from termcolor import colored
from tqdm import tqdm
import sys
import os

def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
    import time
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_{dist_rank}_{formatted_time}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get wikitext2")
    traindata = load_dataset("/path_to_datasets/wikitext", 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset("/path_to_datasets/wikitext", 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_ptb(tokenizer, nsamples, val_size, seed, seqlen, test_only):
    print("get_ptb")
    valdata = load_dataset('/path_to_datasets/ptb_text_only/', 'penn_treebank', split='validation',
                           trust_remote_code=True)
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    if test_only:
        return testenc
    traindata = load_dataset('/path_to_datasets/ptb_text_only/', 'penn_treebank', split='train')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    traindata = load_dataset(
        '/path_to_datasets/c4', data_files={'train': '/path_to_datasets/c4/c4-train.00000-of-01024.json'},
        split='train'
    )
    valdata = load_dataset(
        '/path_to_datasets/c4',
        data_files={'validation': '/path_to_datasets/c4/c4-validation.00000-of-00008.json'}, split='validation'
    )

    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'c4-train.00000-of-01024.json'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'c4-validation.00000-of-00008.json'},
    #     split='validation'
    # )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader


def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")
    try:
        loacal_dataset = "/path_to_datasets/RedPajama-Data-1T-Sample"
        traindata = load_dataset(loacal_dataset, split='train')
    except:
        traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train')
        exit()
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader

def get_loaders(
        name, tokenizer, train_size=128, val_size=64, seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'c4' in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'ptb' in name:
        return get_ptb(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer, train_size, val_size, seed, seqlen)
    else:
        raise NotImplementedError


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        nlls = []
        if hasattr(model,'lm_head'): # and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
    return results


@torch.no_grad()
def evaluate(model, tokenizer, logger, save_log_dir, eval_ppl=False, eval_mmlu=False, eval_tasks="", eval_batch_size=1, max_memory='30GiB'):
    '''
    Note: evaluation simply move model to single GPU.
    '''
    # import pdb;pdb.set_trace()
    # block_class_name = model.model.layers[0].__class__.__name__
    # from accelerate import infer_auto_device_map, dispatch_model
    # device_map = infer_auto_device_map(model, max_memory={i: max_memory for i in range(torch.cuda.device_count())})
    # model = dispatch_model(model, device_map=device_map)
    results = {}

    if eval_mmlu:
        mmlu_num_few_shots = [5]
        for num_few_shots in mmlu_num_few_shots:
            save_dir = os.path.join(save_log_dir, f"mmlu-{num_few_shots}-shot")
            print(save_dir)
            run_mmlu_eval(model, tokenizer, "",
                          num_few_shots, "/path_to_datasets/mmlu_no_train/data/", save_dir)

    if eval_ppl:
        datasets = ["wikitext2","c4","ptb"] #"c4",,"c4","ptb""ptb"["wikitext2"] #
        ppl_results = test_ppl(model, tokenizer, datasets, 2048)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = eval_tasks.split(',')
        model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)

        from lm_eval import utils as lm_eval_utils
        task_manager = lm_eval.tasks.TaskManager(include_path="/path_to_datasets/lm_eval_configs/tasks/",
                                                 include_defaults=False)

        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager
        )
        res_tab = make_table(results)
        logger.info(res_tab)
        print(res_tab)
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')


    return results