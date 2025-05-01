import os
import argparse

def write_yaml(config, path):
    lines = []
    for section, params in config.items():
        lines.append(section)
        for key, val in params.items():
            lines.append(f"{key}: {val}")
        lines.append("")  # 空行分隔
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✔ Saved: {path}")

def generate_configs(model_name, learning_rate, epochs, temperature=0.7, save_dir="./yaml_configs"):
    base_model_path = f"/data/qq/models/Qwen/{model_name}"
    save_prefix = f"saves/{model_name}/full/math-merge"
    lr_str = f"{learning_rate:.0e}"

    # ========== BASE CONFIG ==========
    base_config = {
        "### model": {
            "model_name_or_path": base_model_path,
            "# adapter_name_or_path": "saves/llama3-8b/lora/sft",
            "# trust_remote_code": True
        },
        "### method": {
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "deepspeed": "examples/deepspeed/ds_z3_config.json"
        },
        "### dataset": {
            "dataset": "math-merge",
            "template": "qwen_box",
            "cutoff_len": 2048,
            "max_samples": 100000,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16
        },
        "### output": {
            "output_dir": f"{save_prefix}/base/{lr_str}",
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "overwrite_output_dir": True
        },
        "### train": {
            "per_device_train_batch_size": 1,
            "# gradient_accumulation_steps": "",
            "learning_rate": learning_rate,
            "num_train_epochs": epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000
        },
        "### eval": {
            "# val_size": 0.1,
            "# per_device_eval_batch_size": 1,
            "# eval_strategy": "steps",
            "# eval_steps": 500
        }
    }
    write_yaml(base_config, os.path.join(save_dir, "base.yaml"))

    # ========== STAGE CONFIGS ==========
    prev_model_path = base_model_path
    for stage in range(1, 4):
        dataset = f"math-merge_{model_name}_stage_{stage}_tem_{temperature}"
        output_dir = f"{save_prefix}/stage_{stage}_tem_{temperature}/{lr_str}"
        if stage > 1:
            prev_model_path = f"/data/qq/LLaMA-Factory/{save_prefix}/stage_{stage - 1}_tem_{temperature}/{lr_str}"

        stage_config = {
            "### model": {
                "model_name_or_path": prev_model_path,
                "# adapter_name_or_path": "saves/llama3-8b/lora/sft",
                "# trust_remote_code": True
            },
            "### method": base_config["### method"],
            "### dataset": {
                "dataset": dataset,
                "template": "qwen_box",
                "cutoff_len": 2048,
                "max_samples": 100000,
                "overwrite_cache": True,
                "preprocessing_num_workers": 16
            },
            "### output": {
                "output_dir": output_dir,
                "logging_steps": 10,
                "save_steps": 500,
                "plot_loss": True,
                "overwrite_output_dir": True
            },
            "### train": base_config["### train"],
            "### eval": base_config["### eval"]
        }

        write_yaml(stage_config, os.path.join(save_dir, f"stage_{stage}.yaml"))

# ========== CLI ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base + 3 stage YAML configs.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. Qwen2.5-Math-7B)")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate (e.g. 5e-7)")
    parser.add_argument("--epochs", type=float, required=True, help="Number of training epochs (e.g. 2.0)")
    parser.add_argument("--save_dir", type=str, default="./yaml_configs", help="Directory to save YAML files")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature suffix for dataset/output")

    args = parser.parse_args()

    generate_configs(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        temperature=args.temperature,
        save_dir=args.save_dir
    )
