import runpod
import os
import yaml
import subprocess
import json
import requests
import sys
from pathlib import Path

def maybe_download_dataset(raw_path: str) -> dict:
    """
    If raw_path is a URL, download it to /workspace/data/.
    If it's a HuggingFace dataset ID (no protocol), use as-is.
    Returns a dataset dict for Axolotl config.
    """
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        os.makedirs("/workspace/data", exist_ok=True)
        local_path = "/workspace/data/training_dataset.jsonl"
        print(f"Downloading dataset from {raw_path} ...")
        r = requests.get(raw_path, timeout=300)
        r.raise_for_status()
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"Saved dataset to {local_path} ({len(r.text)} chars)")
        return {"path": local_path, "ds_type": "json", "type": "alpaca"}
    else:
        # Treat as HuggingFace dataset ID or local path
        return {"path": raw_path, "type": "alpaca"}

def handler(event):
    print(f"=== Received input ===")
    print(json.dumps(event, indent=2))

    # Extract input (RunPod wraps in 'input' for /run endpoint)
    input_data = event.get("input", event)
    args = input_data.get("args", input_data)

    # Required fields
    base_model = args.get("base_model", "mistralai/Mistral-7B-v0.1")
    dataset_path = args.get("dataset_path") or args.get("dataset_url")
    
    # Fallback: the app sends datasets array under args
    datasets_config = args.get("datasets", [])
    if not datasets_config and not dataset_path:
        dataset_path = input_data.get("dataset_path") or input_data.get("dataset_url")

    # Build Axolotl dataset config
    if not datasets_config:
        if not dataset_path:
            return {"error": "No dataset_path or datasets provided."}
        datasets_config = [maybe_download_dataset(dataset_path)]

    num_epochs = args.get("num_epochs", 3)
    lora_r = args.get("lora_r", 16)
    output_name = args.get("output_name", "fine-tuned-model")
    gradient_accumulation = args.get("gradient_accumulation_steps", 4)
    micro_batch = args.get("micro_batch_size", 1)
    seq_len = args.get("sequence_len", 4096)

    # Build Axolotl config
    config = {
        "base_model": base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "load_in_4bit": True,
        "adapter": "qlora",
        "lora_r": lora_r,
        "lora_alpha": lora_r * 2,
        "lora_dropout": 0.05,
        "lora_target_linear": True,
        "sequence_len": seq_len,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "datasets": datasets_config,
        "num_epochs": num_epochs,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": 0.0002,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "output_dir": f"./outputs/{output_name}",
        "logging_steps": 10,
        "save_steps": 100,
        "warmup_steps": 100,
        "flash_attention": True,
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "weight_decay": 0.0,
        "special_tokens": {"pad_token": "<|endoftext|>"},
        "max_steps": num_epochs * 100,
    }

    config_path = "/workspace/axolotl_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"=== Wrote config to {config_path} ===")
    print(open(config_path).read())

    # Preprocess
    try:
        prep = subprocess.run(
            ["python", "-m", "axolotl.cli.preprocess", config_path, "--debug-num-examples", "0"],
            capture_output=True, text=True, cwd="/workspace"
        )
        print("=== PREPROCESS STDOUT ===")
        print(prep.stdout)
        if prep.stderr:
            print("=== PREPROCESS STDERR ===")
            print(prep.stderr)
        if prep.returncode != 0:
            return {"status": "error", "stage": "preprocess", "stderr": prep.stderr, "stdout": prep.stdout}
    except Exception as e:
        return {"status": "error", "stage": "preprocess_exception", "error": str(e)}

    # Train
    try:
        train = subprocess.run(
            ["accelerate", "launch", "-m", "axolotl.cli.train", config_path, "--debug-num-examples", "0"],
            capture_output=True, text=True, cwd="/workspace"
        )
        print("=== TRAIN STDOUT (last 2000 chars) ===")
        print(train.stdout[-2000:])
        if train.stderr:
            print("=== TRAIN STDERR (last 2000 chars) ===")
            print(train.stderr[-2000:])
        return {
            "status": "success" if train.returncode == 0 else "error",
            "output_dir": f"/workspace/outputs/{output_name}",
            "stdout": train.stdout[-5000:],
            "stderr": train.stderr[-5000:]
        }
    except Exception as e:
        return {"status": "error", "stage": "train_exception", "error": str(e)}


runpod.serverless.start({"handler": handler})
