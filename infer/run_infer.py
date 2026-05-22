from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infer.pipeline import InferPipeline
from synthetic_icl.backbone import MLLMBackbone


def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file root must be a JSON object.")
    return data


def _coalesce(cli_value: Any, cfg: dict[str, Any], key: str) -> Any:
    return cli_value if cli_value is not None else cfg.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream inference with optional synthetic ICL context.")
    parser.add_argument("--config", help="Path to config JSON.")
    parser.add_argument("--synthetic-output-dir", help="Path to synthetic output cases (0,1,2,...).")
    parser.add_argument("--output-json", help="Path to save inference comparison JSON.")
    parser.add_argument("--per-case-output-dir", help="Directory to save one JSON file per case.")
    parser.add_argument("--infer-api-key", help="Override infer MLLM API key.")
    parser.add_argument("--infer-base-url", help="Override infer MLLM base URL.")
    parser.add_argument("--infer-model-name", help="Override infer MLLM model name.")
    args = parser.parse_args()

    config = _load_config(args.config)
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    infer_cfg = config.get("infer_mllm_api", {}) if isinstance(config.get("infer_mllm_api"), dict) else {}

    synthetic_output_dir = Path(_coalesce(args.synthetic_output_dir, run_cfg, "output_dir") or "synthetic_outputs")
    output_json = Path(_coalesce(args.output_json, run_cfg, "infer_output_json") or "infer/infer_results.json")
    per_case_output_dir = Path(
        _coalesce(args.per_case_output_dir, run_cfg, "infer_per_case_output_dir")
        or (output_json.parent / f"{output_json.stem}_cases")
    )

    backbone = MLLMBackbone(
        api_key=_coalesce(args.infer_api_key, infer_cfg, "api_key"),
        base_url=_coalesce(args.infer_base_url, infer_cfg, "base_url"),
        model=_coalesce(args.infer_model_name, infer_cfg, "model_name") or "gemini-2.5-flash",
    )

    pipeline = InferPipeline(backbone)
    pipeline.run(synthetic_output_dir, output_json, per_case_output_dir=per_case_output_dir)
    print(f"Saved inference comparison to: {output_json}")
    print(f"Saved per-case inference files to: {per_case_output_dir}")


if __name__ == "__main__":
    main()
