MODEL_NAME="unsloth/Llama-3.2-1B"
CALIBRATION_DATASET=fr_awq_dataset
python calibrate.py \
    --model-name ${MODEL_NAME} \
    --max-seq-len 2048 \
    --num-samples 128 \
    --dataset melhoushi/${CALIBRATION_DATASET} \
    --log-dir ./calibrations/${MODEL_NAME}/${CALIBRATION_DATASET}

python eval.py \
    --model-name ${MODEL_NAME} \
    --tasks wikipedia-en wikipedia-sw wikipedia-fr wikipedia-xh wikipedia-st wikipedia-yo wikipedia-zu wikipedia-ha \
    --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=./calibrations/${MODEL_NAME}/${CALIBRATION_DATASET}/${CALIBRATION_DATASET}.pt,scale_sample_weight=True \
    --log-dir ./evals/${MODEL_NAME}/quantized_4bit_${CALIBRATION_DATASET}
    