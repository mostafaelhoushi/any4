#!/bin/bash

# ================================
# Config
# ================================
slurm_cluster="gpu-a10"
slurm_partition="a10"

model_names=(
    "melhoushi/slimpj_h896_d17_gbs182_tpp256.0_lp0.0_uniform"
    "melhoushi/slimpj_h896_d17_gbs182_tpp256.0_lp0.2_linear_linear_reverse"
    "melhoushi/slimpj_h1152_d23_gbs124_tpp64.0_lp0.0_uniform"
    "melhoushi/slimpj_h1152_d23_gbs124_tpp64.0_lp0.4_linear_linear_reverse"


    "melhoushi/slimpj_h1536_d30_gbs136_tpp20.0_lp0.0_uniform"
    "melhoushi/slimpj_h1536_d30_gbs136_tpp20.0_warmup602_lp0.6_linear_linear_reverse"
    "melhoushi/slimpj_h2048_d40_gbs200_tpp30.0_lp0.0_uniform"
    "melhoushi/slimpj_h2048_d40_gbs200_tpp30.0_warmup602_lp0.8_linear_linear_reverse"

    # Models With Same TPP
    "melhoushi/slimpj_h640_d13_nh10_dhqk64_dhv64_ffnmult8_gbs48_tpp20.0_13773steps_dmup1215_depthsupar_pol1_v2"
    "melhoushi/slimpj_h640_d13_gbs48_tpp20.0_lp0.2_linear"
    "melhoushi/slimpj_h640_d13_gbs48_tpp20.0_lp0.2_linear_linear_reverse"
    "melhoushi/slimpj_h896_d17_gbs66_tpp20.0"

    "melhoushi/slimpj_h896_d17_gbs66_tpp20.0_lp0.2_linear_linear_reverse"

    "melhoushi/slimpj_h896_d17_gbs66_tpp20.0_lp0.4_linear_linear_reverse"
    "melhoushi/slimpj_h1152_d23_gbs76_tpp20.0"

    "melhoushi/slimpj_h1152_d23_gbs76_tpp20.0_lp0.2_linear_linear_reverse"

    "melhoushi/slimpj_h1152_d23_gbs76_tpp20.0_lp0.4_linear_linear_reverse"


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
        --tasks piqa arc_easy arc_challenge hellaswag winogrande bbh lambada openbookqa race social_iqa wikitext boolq copa squadv2 wikitext-2 wikipedia c4 c4_new codeparrot \
        --no-overwrite-results --append-results \
        --log-dir ${out_dir}"

    echo \"Launching eval: ${config_name}\"

    cbrun srund \
        -t "${slurm_cluster}" \
        -x "-p ${slurm_partition} -c 4 -J ${config_name} -o ${out_dir}/slurm-%j.out --time 10-0 --wckey sparse_scaling_law" \
        -e "${bash_cmd}"
done
