#!/bin/bash

# ================================
# Config
# ================================
slurm_cluster="gpu-a10"
slurm_partition="a10"

model_names=(
    "unsloth/Llama-3.2-1B"
    "Qwen/Qwen2.5-1.5B"
)

calib_datasets=(
    "en_awq_dataset"
    "sw_awq_dataset"
    "fr_awq_dataset"
    "xh_awq_dataset"
    "zh_awq_dataset"
    "multilingual_awq_dataset"
    "multilingual_awq_dataset_codes"
    "multilingual_awq_dataset_codesmath"
    "multilingual_awq_dataset_math"
    "multilingual_mix_c4_awq_dataset"
    "multilingual_mix_c4_awq_dataset_codes"
    "multilingual_mix_c4_awq_dataset_codesmath"
    "multilingual_mix_c4_awq_dataset_math"
    "multilingual10_awq_dataset"
    "multilingual10_awq_dataset_codes"
    "multilingual10_awq_dataset_codesmath"
    "multilingual10_awq_dataset_math"
)

# ================================
# Loop
# ================================

# 16-bit Float Baseline
for MODEL_NAME in "${model_names[@]}"; do
    config_name="${MODEL_NAME}/eval_${CALIBRATION_DATASET}"
    out_dir="./evals/${MODEL_NAME}/baseline_16bit"

    mkdir -p "${out_dir}"

    bash_cmd="python eval.py \
        --model-name ${MODEL_NAME} \
        --tasks wikipedia-en wikipedia-sw wikipedia-fr wikipedia-xh wikipedia-st wikipedia-yo wikipedia-zu wikipedia-ha wikipedia-ib \
        --no-overwrite-results --append-results \
        --log-dir ${out_dir}"

    echo \"Launching eval: ${config_name}\"

    cbrun srund \
        -t "${slurm_cluster}" \
        -x "-p ${slurm_partition} -c 4 -J ${config_name} -o ${out_dir}/slurm-%j.out --time 10-0 --wckey sparse_scaling_law" \
        -e "${bash_cmd}"
done


# Quantize with No calibration
for MODEL_NAME in "${model_names[@]}"; do
    config_name="${MODEL_NAME}/eval_${CALIBRATION_DATASET}"
    out_dir="./evals/${MODEL_NAME}/quantized_4bit"

    mkdir -p "${out_dir}"

    bash_cmd="python eval.py \
        --model-name ${MODEL_NAME} \
        --tasks wikipedia-en wikipedia-sw wikipedia-fr wikipedia-xh wikipedia-st wikipedia-yo wikipedia-zu wikipedia-ha wikipedia-ib \
        --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head \
        --no-overwrite-results --append-results \
        --log-dir ${out_dir}"

    echo \"Launching eval: ${config_name}\"

    cbrun srund \
        -t "${slurm_cluster}" \
        -x "-p ${slurm_partition} -c 4 -J ${config_name} -o ${out_dir}/slurm-%j.out --time 10-0 --wckey sparse_scaling_law" \
        -e "${bash_cmd}"
done

# Quantize With calibration
for MODEL_NAME in "${model_names[@]}"; do
    for CALIBRATION_DATASET in "${calib_datasets[@]}"; do
        
        config_name="${MODEL_NAME}/eval_${CALIBRATION_DATASET}"
        out_dir="./evals/${MODEL_NAME}/quantized_4bit_${CALIBRATION_DATASET}"
        calib_weight="./calibrations/${MODEL_NAME}/${CALIBRATION_DATASET}/${CALIBRATION_DATASET}.pt"

        mkdir -p "${out_dir}"

        bash_cmd="python eval.py \
            --model-name ${MODEL_NAME} \
            --tasks wikipedia-en wikipedia-sw wikipedia-fr wikipedia-xh wikipedia-st wikipedia-yo wikipedia-zu wikipedia-ha wikipedia-ib \
            --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=${calib_weight},scale_sample_weight=True \
            --no-overwrite-results --append-results \
            --log-dir ${out_dir}"

        echo \"Launching eval: ${config_name}\"

        cbrun srund \
            -t "${slurm_cluster}" \
            -x "-p ${slurm_partition} -c 4 -J ${config_name} -o ${out_dir}/slurm-%j.out --time 10-0 --wckey sparse_scaling_law" \
            -e "${bash_cmd}"
    done
done
