from datasets import load_dataset
import random

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
    # valloader = []
    # for _ in range(val_size):
    #     i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
    #                        trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     valloader.append((inp, tar))
    return trainloader #, valloader