import os
import json
import pandas as pd

# ---------------------------------------------
# USER CONFIG
# ---------------------------------------------

model_names = [
    "unsloth/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
]

calib_datasets = [
    "en_awq_dataset",
    "sw_awq_dataset",
    "fr_awq_dataset",
    "xh_awq_dataset",
    "zh_awq_dataset",
    "multilingual_awq_dataset",
    "multilingual_awq_dataset_codes",
    "multilingual_awq_dataset_codesmath",
    "multilingual_awq_dataset_math",
    "multilingual_mix_c4_awq_dataset",
    "multilingual_mix_c4_awq_dataset_codes",
    "multilingual_mix_c4_awq_dataset_codesmath",
    "multilingual_mix_c4_awq_dataset_math",
    "multilingual10_awq_dataset",
    "multilingual10_awq_dataset_codes",
    "multilingual10_awq_dataset_codesmath",
    "multilingual10_awq_dataset_math",
]

task_map = {
    "wikipedia-en": "en",
    "wikipedia-sw": "sw",
    "wikipedia-fr": "fr",
    "wikipedia-xh": "xh",
    "wikipedia-st": "st",
    "wikipedia-yo": "yo",
    "wikipedia-zu": "zu",
    "wikipedia-ha": "ha",
    "wikipedia-ib": "ib",
}

eval_tasks = list(task_map.keys())


# ---------------------------------------------
# Safe loader: returns {} if file missing or broken
# ---------------------------------------------
def load_results(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------
# Build table for each model
# ---------------------------------------------
for model in model_names:
    print(f"Processing model: {model}")

    rows = []

    def add_row(quantization, calibration, json_path):
        """Adds a row to 'rows', even if results.json missing."""
        result = load_results(json_path)

        row = {
            "Quantization": quantization,
            "Calibration": calibration,
        }

        # Fill each task column; missing keys become empty cells
        for task in eval_tasks:
            col = task_map[task]
            row[col] = result.get(task, None)

        rows.append(row)

    # -------------------------
    # FP16 baseline
    # -------------------------
    add_row(
        quantization="FP16",
        calibration="None",
        json_path=f"./evals/{model}/baseline_16bit/results.json",
    )

    # -------------------------
    # 4-bit without calibration
    # -------------------------
    add_row(
        quantization="4-bit",
        calibration="None",
        json_path=f"./evals/{model}/quantized_4bit/results.json",
    )

    # -------------------------
    # 4-bit calibrated
    # -------------------------
    for calib in calib_datasets:
        add_row(
            quantization="4-bit",
            calibration=calib,
            json_path=f"./evals/{model}/quantized_4bit_{calib}/results.json",
        )

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Ensure column order
    df = df[["Quantization", "Calibration"] + list(task_map.values())]

    # Output
    out_csv = f"./evals/{model}/results.csv"

    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}\n")
