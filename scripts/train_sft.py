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
import subprocess, sys, os, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.utils.config import get_config


def render_dataset_info(cfg) -> str:
    """Render configs/dataset_info.json with an absolute file_name.

    LLaMA-Factory reads dataset_info.json verbatim — no env interpolation — so
    a SEARCH_ZERO_ROOT outside the repo has to be baked in before training.
    The rendered copy lands in configs/generated/ and is passed via
    --dataset_dir, leaving the checked-in template repo-relative.
    """
    src = os.path.join(ROOT, "configs", "dataset_info.json")
    gen_dir = os.path.join(ROOT, "configs", "generated")
    os.makedirs(gen_dir, exist_ok=True)

    with open(src, "r", encoding="utf-8") as f:
        info = json.load(f)

    for name, spec in info.items():
        fname = spec.get("file_name", "")
        # Leave http(s) URLs and already-absolute paths alone.
        if fname and not os.path.isabs(fname) and "://" not in fname:
            spec["file_name"] = os.path.join(cfg.data_dir, fname)

    dst = os.path.join(gen_dir, "dataset_info.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return gen_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Show command only")
    args = parser.parse_args()

    cfg = get_config()
    data_path = cfg.sft_filtered_path
    if not os.path.exists(data_path):
        print(f"ERROR: SFT data not found at {data_path}")
        print("Run first:")
        print(f"  python scripts/generate_sft_data.py {cfg.hotpotqa_dev_path} 1000 -w 16")
        print(f"  python scripts/filter_sft_data.py -w 16")
        sys.exit(1)

    config_path = os.path.join(ROOT, "configs", "sft_lora.yaml")
    dataset_dir = render_dataset_info(cfg)
    model = args.model or cfg.base_model

    cmd = [
        "llamafactory-cli", "train", config_path,
        "--model_name_or_path", model,
        "--num_train_epochs", str(args.epochs),
        "--dataset_dir", dataset_dir,
        "--output_dir", cfg.sft_checkpoint,
    ]

    print("=" * 60)
    print("Search-R1 SFT Training")
    print("=" * 60)
    print(f"Root:    {cfg.root_dir}")
    print(f"Data:    {data_path}")
    print(f"Model:   {model}")
    print(f"Epochs:  {args.epochs}")
    print(f"Output:  {cfg.sft_checkpoint}")
    print(f"Method:  LoRA (rank=16, alpha=32)")
    print(f"Command:\n  {' '.join(cmd)}")
    print("=" * 60)

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
