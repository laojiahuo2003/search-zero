"""
Quick SFT training launcher for Search-R1 agent data.

Prerequisites:
    - GPU with >= 16GB VRAM (e.g., A100, 4090)
    - Generated SFT data at data/sft/sft_trajectories.jsonl

Usage:
    # Full LoRA fine-tune
    llamafactory-cli train configs/sft_lora.yaml

    # Or use this script
    python scripts/train_sft.py

    # Quick test: train on smaller model first
    python scripts/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
"""
import subprocess, sys, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Show command only")
    args = parser.parse_args()

    data_path = os.path.join(ROOT, "data", "sft", "sft_trajectories.jsonl")
    if not os.path.exists(data_path):
        print(f"ERROR: SFT data not found at {data_path}")
        print("Run first: python scripts/generate_sft_data.py data/hotpotqa_dev.json 1000 -w 16")
        sys.exit(1)

    config_path = os.path.join(ROOT, "configs", "sft_lora.yaml")

    cmd = [
        "llamafactory-cli", "train", config_path,
        "--model_name_or_path", args.model,
        "--num_train_epochs", str(args.epochs),
    ]

    print("=" * 60)
    print("Search-R1 SFT Training")
    print("=" * 60)
    print(f"Data:    {data_path}")
    print(f"Model:   {args.model}")
    print(f"Epochs:  {args.epochs}")
    print(f"Method:  LoRA (rank=16, alpha=32)")
    print(f"Command:\n  {' '.join(cmd)}")
    print("=" * 60)

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
