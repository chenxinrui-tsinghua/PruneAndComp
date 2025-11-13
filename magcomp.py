import torch
from tqdm import tqdm

def get_scale_params(model, trainloader, i, j, device, num_sample=128):
    model = model.to(device)
    num_layers = len(model.model.layers)
    d_clip = [torch.zeros(1).to(device) for _ in range(num_layers)]
    scale_params = torch.zeros(1).to(device)
    def hook(module, input, output, layer_name):
        d_clip[layer_name] = input[0]
    handles = []

    for l, layer in enumerate(model.model.layers):
        if l==i or l==j:
            handle = layer.register_forward_hook(lambda module, input, output, layer_name=l: hook(module, input, output, layer_name))
            handles.append(handle)

    num_samples = num_sample
    calibration_loop = tqdm(enumerate(trainloader, start=0), desc="Calibrating", total=num_samples)
    for num, batch in calibration_loop:
        batch = batch[0].to(device)
        try:
            with torch.no_grad():
                output = model(batch)
        except IndexError:
            pass
        scale_param = (d_clip[j].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True) /
            d_clip[i].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True))  # [1,1,4096]
        scale_params = scale_params + scale_param

        num_sample -= 1
        if not num_sample:
            break

    scale_params = scale_params / num_samples
    scale_params = scale_params.mean()

    for handle in handles:
        handle.remove()

    torch.cuda.empty_cache()
    return scale_params

def fuse_scale(model, scale, num):
    model.model.embed_tokens.weight.data *= scale
    for l, layer in enumerate(model.model.layers):
        if l < num:
            layer.self_attn.o_proj.weight.data *= scale
            layer.mlp.down_proj.weight.data *= scale