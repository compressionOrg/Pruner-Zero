"""自动确定alpha参数的模块

基于困惑度使用二分查找法自动确定DLP方法中的最优alpha值。
支持两阶段搜索：快速搜索阶段使用少量样本，最终验证阶段使用完整数据集。
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable
from lib.eval import eval_ppl
from lib.data import get_loaders
import copy
import gc
from pdb import set_trace as st


def eval_ppl_fast(model, tokenizer, device=torch.device("cuda:0"), 
                  dataset="wikitext2", max_samples=10):
    """
    快速困惑度评估函数，使用限定数量的样本进行评估。
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        device: 计算设备
        dataset: 数据集名称
        max_samples: 最大样本数量
        
    Returns:
        float: 困惑度值
    """
    print(f"快速评估 {dataset} (使用 {max_samples} 个样本)")
    
    # 获取测试数据加载器
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )
    
    # 使用限定样本数量进行评估
    with torch.no_grad():
        ppl = eval_ppl_dataset_fast(model, testloader, max_samples=max_samples, device=device)
    return ppl


def eval_ppl_dataset_fast(model, testenc, max_samples=10, bs=1, device=None):
    """
    快速数据集困惑度评估函数，限制样本数量。
    
    Args:
        model: 要评估的模型
        testenc: 测试数据编码
        max_samples: 最大样本数量
        bs: 批次大小
        device: 计算设备
        
    Returns:
        float: 困惑度值
    """
    import torch.nn as nn
    
    # 获取输入ID
    testenc = testenc.input_ids
    
    # 计算样本数量，限制为max_samples
    total_samples = testenc.numel() // model.seqlen
    nsamples = min(total_samples, max_samples)
    
    # 存储负对数似然
    nlls = []
    print(f"快速评估使用 {nsamples}/{total_samples} 个样本")
    
    # 遍历每个批次
    for i in range(0, nsamples, bs):
        if i % 10 == 0:
            print(f"快速评估样本 {i}/{nsamples}")
        
        # 计算结束索引
        j = min(i + bs, nsamples)
        
        # 准备输入并移动到设备
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)
        
        # 前向传播
        lm_logits = model(inputs).logits
        
        # 为下一个token预测移位logits和标签
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        
        # 计算损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                       shift_labels.reshape(-1))
        
        # 计算负对数似然
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        
        # 添加到负对数似然列表
        nlls.append(neg_log_likelihood)
    
    print('开始计算快速PPL')
    # 计算困惑度
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()


# def binary_search_alpha(
#     args,
#     model,
#     tokenizer,
#     prune_func: Callable,
#     device: torch.device = torch.device("cuda:0"),
#     alpha_min: float = 0.01,
#     alpha_max: float = 0.5,
#     tolerance: float = 0.01,
#     max_iterations: int = 10,
#     target_ppl_threshold: Optional[float] = None,
#     fast_search_samples: int = 10,
#     final_validation: bool = True
# ) -> Tuple[float, float]:
#     """
#     使用二分查找法自动确定最优的alpha值。
    
#     Args:
#         args: 包含模型配置的参数对象
#         model: 要剪枝的模型
#         tokenizer: 分词器
#         device: 计算设备
#         alpha_min: alpha的最小搜索范围
#         alpha_max: alpha的最大搜索范围
#         tolerance: 收敛容差
#         max_iterations: 最大迭代次数
#         target_ppl_threshold: 目标困惑度阈值（可选）
        
#     Returns:
#         Tuple[float, float]: (最优alpha值, 对应的困惑度)
#     """
#     print(f"开始自动确定alpha值，搜索范围: [{alpha_min}, {alpha_max}]")
    
#     # 保存原始alpha值
#     original_alpha = args.alpha
    
#     # 存储已测试的alpha值和对应的困惑度
#     tested_alphas = {}
    
#     def evaluate_alpha(alpha_val: float, use_fast_search: bool = True) -> float:
#         """评估给定alpha值的困惑度"""
#         cache_key = f"{alpha_val}_{use_fast_search}"
#         if cache_key in tested_alphas:
#             return tested_alphas[cache_key]
            
#         search_type = "快速搜索" if use_fast_search else "完整验证"
#         print(f"{search_type} alpha = {alpha_val:.4f}")
        
#         # 创建模型副本以避免修改原始模型
#         model_copy = copy.deepcopy(model)
#         args_copy = copy.deepcopy(args)
#         args_copy.alpha = alpha_val
        
#         try:
#             # 执行剪枝
#             prune_func(args_copy, model_copy, tokenizer, device)
            
#             # 评估困惑度
#             if use_fast_search:
#                 # 快速搜索：使用少量样本
#                 ppl = eval_ppl_fast(model_copy, tokenizer, device, 
#                                   dataset=args.dataset, max_samples=fast_search_samples)
#             else:
#                 # 完整验证：使用全部数据集
#                 ppl = eval_ppl(model_copy, tokenizer, device, dataset=args.dataset)
            
#             tested_alphas[cache_key] = ppl
#             print(f"alpha = {alpha_val:.4f}, PPL = {ppl:.4f} ({search_type})")
            
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
            
#             return ppl
            
#         except Exception as e:
#             print(f"评估alpha = {alpha_val:.4f}时出错: {e}")
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
#             return float('inf')
    
#     # 初始化搜索边界
#     left, right = alpha_min, alpha_max
#     best_alpha = original_alpha
#     best_ppl = float('inf')
    
#     # 第一阶段：使用少量样本快速搜索最优alpha
#     print(f"\n=== 第一阶段：快速搜索 (使用 {fast_search_samples} 个样本) ===")
    
#     # 首先评估边界值
#     ppl_left = evaluate_alpha(left, use_fast_search=True)
#     ppl_right = evaluate_alpha(right, use_fast_search=True)
    
#     # 更新最佳值
#     if ppl_left < best_ppl:
#         best_alpha, best_ppl = left, ppl_left
#     if ppl_right < best_ppl:
#         best_alpha, best_ppl = right, ppl_right
    
#     iteration = 0
#     while iteration < max_iterations and (right - left) > tolerance:
#         iteration += 1
        
#         # 计算中点
#         mid = (left + right) / 2.0
#         ppl_mid = evaluate_alpha(mid, use_fast_search=True)
        
#         # 更新最佳值
#         if ppl_mid < best_ppl:
#             best_alpha, best_ppl = mid, ppl_mid
        
#         # 如果达到目标困惑度阈值，提前停止
#         if target_ppl_threshold and ppl_mid <= target_ppl_threshold:
#             print(f"达到目标困惑度阈值 {target_ppl_threshold}，提前停止")
#             break
        
#         # 计算左右两个四分位点
#         left_quarter = left + (mid - left) / 2
#         right_quarter = mid + (right - mid) / 2
        
#         ppl_left_quarter = evaluate_alpha(left_quarter, use_fast_search=True)
#         ppl_right_quarter = evaluate_alpha(right_quarter, use_fast_search=True)
        
#         # 更新最佳值
#         if ppl_left_quarter < best_ppl:
#             best_alpha, best_ppl = left_quarter, ppl_left_quarter
#         if ppl_right_quarter < best_ppl:
#             best_alpha, best_ppl = right_quarter, ppl_right_quarter
        
#         # 选择困惑度更低的区间继续搜索
#         if ppl_left_quarter <= ppl_right_quarter:
#             right = mid
#         else:
#             left = mid
        
#         print(f"快速搜索迭代 {iteration}: 搜索区间 [{left:.4f}, {right:.4f}], 当前最佳 alpha = {best_alpha:.4f}, PPL = {best_ppl:.4f}")
    
#     print(f"快速搜索完成: 最优 alpha = {best_alpha:.4f}, 快速PPL = {best_ppl:.4f}")
    
#     # 第二阶段：使用完整数据集验证最优alpha
#     if final_validation:
#         print(f"\n=== 第二阶段：完整数据集验证 ===")
#         final_ppl = evaluate_alpha(best_alpha, use_fast_search=False)
#         print(f"最终验证: alpha = {best_alpha:.4f}, 完整PPL = {final_ppl:.4f}")
#         best_ppl = final_ppl
    
#     # 恢复原始alpha值
#     args.alpha = original_alpha
    
#     print(f"\n自动alpha确定完成: 最优 alpha = {best_alpha:.4f}, 最终PPL = {best_ppl:.4f}")
#     print(f"所有测试结果: {tested_alphas}")
    
#     return best_alpha, best_ppl


# def adaptive_alpha_search(
#     args,
#     model,
#     tokenizer,
#     prune_func: Callable,
#     device: torch.device = torch.device("cuda:0"),
#     initial_alphas: list = [0.05, 0.1, 0.15, 0.2, 0.25],
#     refinement_steps: int = 3,
#     fast_search_samples: int = 10,
#     final_validation: bool = True
# ) -> Tuple[float, float]:
#     """
#     自适应alpha搜索算法，先粗搜索后细搜索。
    
#     Args:
#         args: 包含模型配置的参数对象
#         model: 要剪枝的模型
#         tokenizer: 分词器
#         device: 计算设备
#         initial_alphas: 初始搜索的alpha候选值列表
#         refinement_steps: 细化搜索的步数
        
#     Returns:
#         Tuple[float, float]: (最优alpha值, 对应的困惑度)
#     """
#     print(f"开始自适应alpha搜索，初始候选值: {initial_alphas}")
    
#     # 保存原始alpha值
#     original_alpha = args.alpha
    
#     # 第一阶段：粗搜索
#     best_alpha = original_alpha
#     best_ppl = float('inf')
#     alpha_ppl_pairs = []
    
#     # 第一阶段：使用少量样本进行粗搜索
#     print(f"\n=== 第一阶段：粗搜索 (使用 {fast_search_samples} 个样本) ===")
    
#     for alpha in initial_alphas:
#         print(f"粗搜索测试 alpha = {alpha:.4f}")
        
#         # 创建模型副本
#         model_copy = copy.deepcopy(model)
#         args_copy = copy.deepcopy(args)
#         args_copy.alpha = alpha
        
#         try:
#             # 执行剪枝
#             prune_func(args_copy, model_copy, tokenizer, device)
            
#             # 评估困惑度（使用少量样本）
#             ppl = eval_ppl_fast(model_copy, tokenizer, device, 
#                               dataset=args.dataset, max_samples=fast_search_samples)
            
#             alpha_ppl_pairs.append((alpha, ppl))
#             print(f"alpha = {alpha:.4f}, 快速PPL = {ppl:.4f}")
            
#             if ppl < best_ppl:
#                 best_alpha, best_ppl = alpha, ppl
            
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
            
#         except Exception as e:
#             print(f"评估alpha = {alpha:.4f}时出错: {e}")
#             alpha_ppl_pairs.append((alpha, float('inf')))
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
    
#     # 排序找到最佳的几个alpha值
#     alpha_ppl_pairs.sort(key=lambda x: x[1])
#     print(f"粗搜索结果: {alpha_ppl_pairs}")
    
#     # 第二阶段：在最佳alpha附近进行细搜索
#     if len(alpha_ppl_pairs) >= 2:
#         # 找到最佳alpha的邻域
#         best_coarse_alpha = alpha_ppl_pairs[0][0]
        
#         # 确定细搜索范围
#         alpha_range = 0.05  # 搜索范围
#         alpha_min = max(0.01, best_coarse_alpha - alpha_range)
#         alpha_max = min(0.5, best_coarse_alpha + alpha_range)
        
#         print(f"开始细搜索，范围: [{alpha_min:.4f}, {alpha_max:.4f}]")
        
#         print(f"\n=== 第二阶段：细搜索，范围: [{alpha_min:.4f}, {alpha_max:.4f}] ===")
        
#         # 使用二分查找进行细搜索（仍使用快速搜索）
#         refined_alpha, refined_ppl = binary_search_alpha(
#             args, model, tokenizer, prune_func, device,
#             alpha_min=alpha_min,
#             alpha_max=alpha_max,
#             tolerance=0.005,
#             max_iterations=refinement_steps,
#             fast_search_samples=fast_search_samples,
#             final_validation=False  # 在这里不进行最终验证
#         )
        
#         if refined_ppl < best_ppl:
#             best_alpha, best_ppl = refined_alpha, refined_ppl
    
#     # 第三阶段：使用完整数据集验证最优alpha
#     if final_validation:
#         print(f"\n=== 第三阶段：完整数据集验证 ===")
        
#         # 创建模型副本进行最终验证
#         model_copy = copy.deepcopy(model)
#         args_copy = copy.deepcopy(args)
#         args_copy.alpha = best_alpha
        
#         try:
#             # 执行剪枝
#             prune_func(args_copy, model_copy, tokenizer, device)
            
#             # 使用完整数据集评估困惑度
#             final_ppl = eval_ppl(model_copy, tokenizer, device, dataset=args.dataset)
            
#             print(f"最终验证: alpha = {best_alpha:.4f}, 完整PPL = {final_ppl:.4f}")
#             best_ppl = final_ppl
            
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
            
#         except Exception as e:
#             print(f"最终验证时出错: {e}")
#             # 清理内存
#             del model_copy
#             if 'args_copy' in locals():
#                 del args_copy
#             gc.collect()
#             torch.cuda.empty_cache()
    
#     # 恢复原始alpha值
#     args.alpha = original_alpha
    
#     print(f"\n自适应alpha搜索完成: 最优 alpha = {best_alpha:.4f}, 最终PPL = {best_ppl:.4f}")
    
#     return best_alpha, best_ppl


def ternary_search_alpha(
    args,
    model_loader: Callable,
    tokenizer,
    prune_func: Callable,
    device: torch.device = torch.device("cuda:0"),
    prune_n: int = 0,
    prune_m: int = 0,
    alpha_min: float = 0.0,
    alpha_max: float = 0.3,
    max_iterations: int = 10,
    tolerance: float = 0.05,
    fast_search_samples: int = 10,
    final_validation: bool = True
) -> Tuple[float, float]:
    """
    基于三分搜索法（Ternary Search）自动确定最优的alpha值，对应HSP算法伪代码。
    
    Args:
        args: 包含模型配置的参数对象
        model_loader: 模型加载函数，用于每次评估时重新加载全新模型以节省显存
        tokenizer: 分词器
        prune_func: 剪枝函数
        device: 计算设备
        alpha_min: alpha的最小搜索范围 (Algorithm 1: \alpha_min)
        alpha_max: alpha的最大搜索范围 (Algorithm 1: \alpha_max)
        max_iterations: 最大迭代次数 T (Algorithm 1: T)
        tolerance: 收敛容差（区间宽度小于该值时提前停止）
        fast_search_samples: 快速评估使用的样本数 (类似 Algorithm 1 中的 D_cal)
        final_validation: 是否在最后使用完整数据集进行验证
        
    Returns:
        Tuple[float, float]: (最优alpha值, 对应的困惑度)
    """
    import os
    import json
    
    # 1. 构建唯一的缓存键（Cache Key）
    model_name = args.model_name if hasattr(args, 'model_name') else 'unknown_model'
    sparsity = getattr(args, 'sparsity_ratio', 'unknown_sparsity')
    method_name = prune_func.__name__ if hasattr(prune_func, '__name__') else 'unknown_method'
    sparsity_type = getattr(args, 'sparsity_type', 'unstructured')
    cache_key = f"{model_name}_sparsity{sparsity}_type{sparsity_type}_{method_name}"
    
    # 2. 检查并读取缓存（Cache Hit）
    cache_file = "./best_alpha_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache_dict = json.load(f)
            if cache_key in cache_dict:
                cached_alpha = cache_dict[cache_key].get("alpha")
                cached_ppl = cache_dict[cache_key].get("ppl")
                print(f"已找到缓存的最佳 alpha: {cached_alpha} (PPL: {cached_ppl})，跳过搜索直接返回。")
                return cached_alpha, cached_ppl
        except Exception as e:
            print(f"读取缓存文件失败: {e}")

    sparsity_mode = "unstructured" if prune_n == 0 else f"{prune_n}:{prune_m}"
    print(
        f"开始三分搜索法确定alpha值，搜索范围: [{alpha_min}, {alpha_max}], "
        f"最大迭代次数: {max_iterations}, 容差: {tolerance}"
    )
    print(
        f"alpha搜索稀疏模式: mode={sparsity_mode}, "
        f"args.sparsity_type={sparsity_type}, prune_n={prune_n}, prune_m={prune_m}"
    )
    
    # 保存原始alpha值
    original_alpha = args.alpha
    
    # 存储已测试的alpha值和对应的困惑度
    tested_alphas = {}
    
    def evaluate_alpha(alpha_val: float, use_fast_search: bool = True) -> float:
        """评估给定alpha值的困惑度"""
        
        cache_key = f"{alpha_val:.6f}_{use_fast_search}"
        if cache_key in tested_alphas:
            return tested_alphas[cache_key]
            
        search_type = "快速搜索" if use_fast_search else "完整验证"
        print(f"{search_type} alpha = {alpha_val:.4f}")
        
        # 释放显存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 重新加载模型以节省显存，避免深拷贝
        print("重新加载模型以进行评估...")
        model_copy = model_loader()
        
        # 确保模型在正确的设备上
        if hasattr(model_copy, 'to') and not hasattr(model_copy, 'hf_device_map'):
            model_copy.to(device)
            
        args_copy = copy.deepcopy(args)
        args_copy.alpha = alpha_val
        
        try:
            # 临时修改 prune_func，防止在搜索过程中保存剪枝后的模型权重
            # 因为多次剪枝会产生大量中间变量，导致 OOM
            if hasattr(args_copy, 'save_model'):
                original_save_model = args_copy.save_model
                args_copy.save_model = None

            # 执行剪枝 (Algorithm 1: Prune(W, {I^l}, p, m))
            prune_func(
                args_copy,
                model_copy,
                tokenizer,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
            )
            
            # 恢复参数
            if hasattr(args_copy, 'save_model'):
                args_copy.save_model = original_save_model
                
            # 评估困惑度 (Algorithm 1: PPL(..., D_cal))
            dataset_name = getattr(args_copy, "dataset", "wikitext2")
            if use_fast_search:
                # 快速搜索：使用少量样本
                ppl = eval_ppl_fast(model_copy, tokenizer, device, 
                                  dataset=dataset_name, max_samples=fast_search_samples)
            else:
                # 完整验证：使用全部数据集
                ppl = eval_ppl(args_copy, model_copy, tokenizer, device=device)
            
            tested_alphas[cache_key] = ppl
            print(f"alpha = {alpha_val:.4f}, PPL = {ppl:.4f} ({search_type})")
            
            # 清理内存
            del model_copy
            if 'args_copy' in locals():
                del args_copy
            gc.collect()
            torch.cuda.empty_cache()
            
            return ppl
            
        except Exception as e:
            print(f"评估alpha = {alpha_val:.4f}时出错: {e}")
            # 清理内存
            if 'model_copy' in locals():
                del model_copy
            if 'args_copy' in locals():
                del args_copy
            gc.collect()
            torch.cuda.empty_cache()
            return float('inf')
            
    # 初始化搜索边界和最佳值 (Algorithm 1, Line 6)
    l = alpha_min
    u = alpha_max
    p_star = float('inf')
    alpha_star = l
    t = 0
    
    print(f"\\n=== 第一阶段：三分搜索 (使用 {fast_search_samples} 个样本) ===")
    
    # 开始迭代 (Algorithm 1, Line 7: while t < T do)
    while t < max_iterations and (u - l) > tolerance:
        # 定义两个内点 (Algorithm 1, Line 8)
        m1 = l + (u - l) / 3.0
        m2 = u - (u - l) / 3.0
        
        print(f"迭代 {t}: 搜索区间 [{l:.4f}, {u:.4f}], m1={m1:.4f}, m2={m2:.4f}")
        
        # 在校准集上评估困惑度 (Algorithm 1, Line 9-11)
        p1 = evaluate_alpha(m1, use_fast_search=True)
        p2 = evaluate_alpha(m2, use_fast_search=True)
        
        # 比较和更新 (Algorithm 1, Line 12-22)
        if p1 < p2:
            u = m2  # 缩小搜索区间到左侧 (Line 13)
            if p1 < p_star:  # (Line 14)
                p_star = p1
                alpha_star = m1  # Update bests (Line 15)
        else:
            l = m1  # 缩小搜索区间到右侧 (Line 18)
            if p2 < p_star:  # (Line 19)
                p_star = p2
                alpha_star = m2  # Update bests (Line 20)
                
        t += 1  # (Line 23)
        print(f"当前最佳 alpha = {alpha_star:.4f}, PPL = {p_star:.4f}")

    print(f"三分搜索完成: 最优 alpha = {alpha_star:.4f}, 快速PPL = {p_star:.4f}")
    
    # 第二阶段：使用完整数据集验证最优alpha (Algorithm 1, Line 25-26)
    if final_validation:
        print(f"\\n=== 第二阶段：完整数据集验证 ===")
        final_ppl = evaluate_alpha(alpha_star, use_fast_search=False)
        print(f"最终验证: alpha = {alpha_star:.4f}, 完整PPL = {final_ppl:.4f}")
        p_star = final_ppl
    
    # 恢复原始alpha值
    args.alpha = original_alpha
    
    print(f"\\n自动alpha三分搜索完成: 最优 alpha = {alpha_star:.4f}, 最终PPL = {p_star:.4f}")
    print(f"所有测试结果: {tested_alphas}")
    
    # 4. 保存结果到缓存（Save Cache）
    try:
        cache_dict = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_dict = json.load(f)
        
        cache_dict[cache_key] = {"alpha": alpha_star, "ppl": p_star}
        
        with open(cache_file, "w") as f:
            json.dump(cache_dict, f, indent=4)
        print(f"已将最优 alpha 保存到缓存文件 {cache_file} 中。")
    except Exception as e:
        print(f"保存缓存文件失败: {e}")
        
    return alpha_star, p_star
