import os
import json
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from typing import Tuple, List


def parse_args() -> Tuple[DictConfig, str, List[str]]:
    parser = ArgumentParser(description="Merge base config with CLI overrides and export a resolved config.")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Base config path (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Resolved config path (.yaml/.yml/.json). Default: <config_file>.resolved.yaml",
    )

    # Keep unknown args for OmegaConf dotlist overrides.
    args, unknown = parser.parse_known_args()

    base_cfg = OmegaConf.load(args.config_file)

    # Default output: base.yaml -> base.resolved.yaml
    if args.output_file is None:
        base_dir = os.path.dirname(os.path.abspath(args.config_file))
        base_name = os.path.basename(args.config_file)
        stem, _ = os.path.splitext(base_name)
        args.output_file = os.path.join(base_dir, f"{stem}.resolved.yaml")

    return base_cfg, args.output_file, unknown


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_config(cfg: DictConfig, output_file: str) -> None:
    _ensure_parent_dir(output_file)
    _, ext = os.path.splitext(output_file.lower())

    if ext in [".yaml", ".yml"]:
        OmegaConf.save(cfg, output_file)
        return

    if ext == ".json":
        # Convert to plain Python types for JSON.
        obj = OmegaConf.to_container(cfg, resolve=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output extension: {ext}. Use .yaml/.yml/.json")


def main():
    base_cfg, output_file, overrides = parse_args()

    # Parse CLI overrides like key=value or a.b.c=xxx.
    cli_cfg = OmegaConf.from_dotlist(overrides)

    # Merge configs; later values override earlier ones.
    merged = OmegaConf.merge(base_cfg, cli_cfg)

    # Resolve interpolations like ${param}.
    OmegaConf.resolve(merged)

    # Write the resolved config file.
    _save_config(merged, output_file)


if __name__ == "__main__":
    main()
