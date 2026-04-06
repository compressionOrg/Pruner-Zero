import time 
import heapq 
import gc
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .auto_alpha import ternary_search_alpha

# from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:     
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device) # ori: 128
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



# @torch.no_grad()
# def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
#     ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
#     print('Starting ...')
#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     if "OPT" in model.__class__.__name__:
#         layers = model.model.decoder.layers 
#     else: 
#         layers = model.model.layers

#     if "model.embed_tokens" in model.hf_device_map:
#         dev = model.hf_device_map["model.embed_tokens"]

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {'i': 0, 'attention_mask': None, "position_ids": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             if "OPT" in model.__class__.__name__:
#                 cache['position_ids'] = None 
#             else:
#                 cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
        
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(dev))
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']

#     print('Ready.')

#     for i in range(len(layers)):
#         layer = layers[i]
#         if f"model.layers.{i}" in model.hf_device_map:
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             print(f"layer {i} device {dev}")
#             inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

#         subset = find_layers(layer)

#         gpts = {}
#         for name in subset:
#             gpts[name] = AblateGPT(subset[name], gradient_path=args.gradient_path)

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 gpts[name].add_batch(inp[0].data, out.data)
#             return tmp

#         handles = []
#         for name in gpts:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))

#         for j in range(args.nsamples):
#             if "OPT" in model.__class__.__name__:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
#             else:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

#         for h in handles:
#             h.remove()

#         for name in gpts:
#             print(i, name)
#             print('Pruning ...')

#             if args.prune_method == "ablate_wanda_seq":
#                 prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
#             elif args.prune_method == "ablate_mag_seq":
#                 prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
#             elif args.prune_method == "ablate_prunerzero_seq":
#                 indexed_name = f'{name}_layer_{i}'
#                 prune_mask = gpts[name].get_prunerzero_mask(args.sparsity_ratio, prune_n, prune_m, indexed_name)
#             elif "iter" in args.prune_method:
#                 prune_mask = None 

#             indexed_name = f'{name}_layer_{i}'
#             gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, indexed_name=indexed_name)
#             gpts[name].free()

#         for j in range(args.nsamples):
#             if "OPT" in model.__class__.__name__:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
#             else:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

#         layers[i] = layer 
#         torch.cuda.empty_cache()

#         inps, outs = outs, inps

#     model.config.use_cache = use_cache
#     torch.cuda.empty_cache()

    
def prune_pruner_zero(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, engine=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"pruning layer {i} name {name}")
            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1,-1))
            G = gradients[indexed_name]
            
            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )
            assert W_metric is not None

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_pruner_zero_dlp(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, engine=None
):
    """
    Pruner-Zero 的按层稀疏率（DLP）版本。
    层间重要性分配逻辑保持与 DSnoT DLP 一致，层内剪枝逻辑使用 pruner-zero。
    """
    if engine is None:
        raise ValueError("prune_pruner_zero_dlp requires an engine for pruner-zero scoring")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))

    layers = model.model.layers
    layer_score = {}

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            indexed_name = f"{name}_layer_{i}"
            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            G = gradients[indexed_name]

            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )
            assert W_metric is not None
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        strategy = getattr(args, "strategy", "mean")
        if strategy == "sum":
            strategy_score = torch.sum(layer_wmetric.float())
        elif strategy == "max":
            strategy_score = torch.max(layer_wmetric.float())
        elif strategy == "median":
            strategy_score = torch.median(layer_wmetric.float())
        elif strategy == "std":
            strategy_score = torch.std(layer_wmetric.float())
        elif strategy == "var":
            strategy_score = torch.var(layer_wmetric.float())
        else:
            strategy_score = torch.mean(layer_wmetric.float())

        layer_score[i] = torch.sum(torch.abs(strategy_score))

    total_conn = sum(layer_score.values())
    ratio_conn = {}
    for layer_idx in layer_score:
        conn_ratio = layer_score[layer_idx] / total_conn
        ratio_conn[layer_idx] = 1 - conn_ratio

    imp_ratios = torch.tensor(list(ratio_conn.values()))
    min_ratio = torch.min(imp_ratios)
    max_ratio = torch.max(imp_ratios)
    alpha = getattr(args, "alpha", 0.5)
    if torch.isclose(max_ratio, min_ratio):
        all_layer_ratio = [1 - args.sparsity_ratio] * len(layers)
    else:
        scaled_ratios = (imp_ratios - min_ratio) * (
            1 / (max_ratio - min_ratio) * alpha * 2
        )
        all_layer_ratio = (
            scaled_ratios - torch.mean(scaled_ratios) + (1 - args.sparsity_ratio)
        ).tolist()

    print("per-layer keep ratio:", all_layer_ratio)

    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        if layer_sparsity_ratio <= 0:
            layer_sparsity_ratio = 0.01
        elif layer_sparsity_ratio >= 1:
            layer_sparsity_ratio = 0.99

        for name in subset:
            print(
                f"pruning layer {i} name {name} with layer sparsity {layer_sparsity_ratio:.4f}"
            )

            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            G = gradients[f"{name}_layer_{i}"]

            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )
            assert W_metric is not None

            W_mask = torch.zeros_like(W_metric) == 1
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                sparsity_num = int(W_metric.shape[1] * layer_sparsity_ratio)
                indices = sort_res[1][:, :sparsity_num]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_pruner_zero_dlp_auto(
    args,
    model_loader,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    engine=None,
):
    """
    带自动 alpha 搜索的 pruner-zero DLP 剪枝方法。
    层间重要性计算与 DLP 保持一致，层内使用 pruner-zero 评分。
    """
    if engine is None:
        raise ValueError("prune_pruner_zero_dlp_auto requires an engine for pruner-zero scoring")

    if not hasattr(args, "alpha"):
        args.alpha = 0.4

    if hasattr(args, "auto_alpha") and args.auto_alpha:
        print("启用自动alpha确定功能 (节省显存模式)")

        alpha_min = getattr(args, "alpha_min", 0.0)
        alpha_max = getattr(args, "alpha_max", 0.3)
        alpha_tolerance = getattr(args, "alpha_tolerance", 0.025)
        alpha_max_iter = getattr(args, "alpha_max_iter", 10)

        def prune_func(alpha_args, alpha_model, alpha_tokenizer, alpha_device, prune_n=0, prune_m=0):
            return prune_pruner_zero_dlp(
                alpha_args,
                alpha_model,
                alpha_tokenizer,
                alpha_device,
                prune_n=prune_n,
                prune_m=prune_m,
                engine=engine,
            )

        optimal_alpha, optimal_ppl = ternary_search_alpha(
            args,
            model_loader,
            tokenizer,
            prune_func=prune_func,
            device=device,
            prune_n=prune_n,
            prune_m=prune_m,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            tolerance=alpha_tolerance,
            max_iterations=alpha_max_iter,
            fast_search_samples=5,
            final_validation=False,
        )

        print(f"自动确定的最优alpha值: {optimal_alpha:.4f}, 记录alpha值并准备最终剪枝")
        args.alpha = optimal_alpha

        print("清掉占用的显存，重新加载模型...")
        gc.collect()
        torch.cuda.empty_cache()

        model = model_loader()

        if "30b" in args.model or "65b" in args.model:
            device = model.hf_device_map["lm_head"]
        elif hasattr(model, "to") and not hasattr(model, "hf_device_map"):
            model.to(device)

        print(f"使用最佳的alpha={optimal_alpha:.4f}进行剪枝...")
        prune_pruner_zero_dlp(
            args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine
        )

        print(f"使用自动确定的alpha={optimal_alpha:.4f}完成剪枝")
        return model
    else:
        print(f"使用手动设置的alpha值: {args.alpha}")
        model = model_loader()
        if "30b" in args.model or "65b" in args.model:
            device = model.hf_device_map["lm_head"]
        elif hasattr(model, "to") and not hasattr(model, "hf_device_map"):
            model.to(device)
        prune_pruner_zero_dlp(
            args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine
        )
        return model

