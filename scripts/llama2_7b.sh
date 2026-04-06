set -x
# conda activate prune
export CUDA_VISIBLE_DEVICES=3

python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method pruner-zero \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --gradient_path  "gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth" \
    --save out/llama2_7b/unstructured/pruner-zero/ \
    --eval_zero_shot \
    2>&1 | tee Llama-2-7b-hf_zeroshot.log


set +x