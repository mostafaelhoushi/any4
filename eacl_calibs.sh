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
for MODEL_NAME in "${model_names[@]}"; do
    for CALIBRATION_DATASET in "${calib_datasets[@]}"; do
        
        config_name="${MODEL_NAME}/${CALIBRATION_DATASET}"
        out_dir="./calibrations/${MODEL_NAME}/${CALIBRATION_DATASET}"

        # Create output directory
        mkdir -p "${out_dir}"

        bash_cmd="python calibrate.py \
            --model-name ${MODEL_NAME} \
            --max-seq-len 2048 \
            --num-samples 128 \
            --dataset melhoushi/${CALIBRATION_DATASET} \
            --log-dir ${out_dir}"

        echo "Launching: ${config_name}"

        cbrun srund \
            -t "${slurm_cluster}" \
            -x "-p ${slurm_partition} -c 4 -J ${config_name} -o ${out_dir}/slurm-%j.out --time 10-0 --wckey sparse_scaling_law" \
            -e "${bash_cmd}"

    done
done
