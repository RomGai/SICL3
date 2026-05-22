from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from synthetic_icl.backbone import MLLMBackbone


@dataclass
class ContextExample:
    image_path: Path
    query: str
    answer: str


class InferPipeline:
    def __init__(self, backbone: MLLMBackbone) -> None:
        self.backbone = backbone

    @staticmethod
    def _load_case_meta(case_dir: Path) -> dict[str, str]:
        meta_path = case_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {case_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "prompt": str(meta.get("prompt", "")).strip(),
            "groundtruth": str(meta.get("groundtruth", "")).strip(),
        }

    @staticmethod
    def _find_gt_image(case_dir: Path) -> Path:
        candidates = sorted(case_dir.glob("*_gt.*"))
        if not candidates:
            raise FileNotFoundError(f"Missing *_gt image in {case_dir}")
        return candidates[0]

    @staticmethod
    def _load_context_examples(case_dir: Path) -> list[ContextExample]:
        run_logs = sorted(case_dir.glob("*run_log*.json")) + sorted(case_dir.glob("run_log.json"))
        if not run_logs:
            return []
        run_log = json.loads(run_logs[0].read_text(encoding="utf-8"))
        selected = run_log.get("selected_examples", []) if isinstance(run_log, dict) else []
        contexts: list[ContextExample] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            vr = item.get("verification_result", {}) if isinstance(item.get("verification_result"), dict) else {}
            saved = vr.get("saved_image_path")
            ans = item.get("answer")
            q = item.get("query")
            if not saved or not ans or not q:
                continue
            image_path = Path(saved)
            if not image_path.exists():
                alt = case_dir / image_path.name
                if alt.exists():
                    image_path = alt
                else:
                    continue
            contexts.append(ContextExample(image_path=image_path, query=str(q), answer=str(ans)))
        return contexts

    def _build_icl_prompt(self, query: str, contexts: list[ContextExample]) -> str:
        lines = [
            "You are a visual question answering assistant.",
            "I will provide context examples first, each with an image and a Q/A pair.",
            "Then I will provide a target image and target question.",
            "First reason based on the context examples and the target image, then answer the final target question.",
            "Output format must be exactly:",
            "Reasoning: <step-by-step reasoning>",
            "Final answer: <concise final answer>",
            "",
        ]
        for idx, ctx in enumerate(contexts, start=1):
            lines.append(f"Context #{idx}: question = {ctx.query}")
            lines.append(f"Context #{idx}: answer = {ctx.answer}")
        lines.append("")
        lines.append(f"Target question: {query}")
        return "\n".join(lines)

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        marker = "Final answer:"
        lines = text.splitlines()
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith(marker):
                return stripped[len(marker):].strip()
        if marker in text:
            return text.rsplit(marker, 1)[1].strip()
        return text.strip()

    def infer_case(self, case_dir: Path) -> dict[str, Any]:
        meta = self._load_case_meta(case_dir)
        gt_path = self._find_gt_image(case_dir)
        contexts = self._load_context_examples(case_dir)

        with Image.open(gt_path) as gt_img:
            gt_image = gt_img.convert("RGB")
            raw_answer = self.backbone.generate_response_multimodal_single(gt_image, meta["prompt"])

        icl_answer = raw_answer
        if contexts:
            images = []
            opened = []
            try:
                for ctx in contexts:
                    im = Image.open(ctx.image_path).convert("RGB")
                    opened.append(im)
                    images.append(im)
                images.append(gt_image)
                icl_prompt = self._build_icl_prompt(meta["prompt"], contexts)
                icl_answer = self.backbone.generate_response_multimodal_multi(images, icl_prompt)
            finally:
                for im in opened:
                    im.close()

        return {
            "case": case_dir.name,
            "query": meta["prompt"],
            "groundtruth": meta["groundtruth"],
            "raw_result": raw_answer,
            "icl_result": icl_answer,
            "icl_final_answer": self._extract_final_answer(icl_answer),
            "num_context_examples": len(contexts),
            "gt_image_path": str(gt_path),
        }

    def run(
        self,
        synthetic_output_dir: Path,
        output_json_path: Path,
        per_case_output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        case_dirs = sorted((p for p in synthetic_output_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        results: list[dict[str, Any]] = []
        if per_case_output_dir is not None:
            per_case_output_dir.mkdir(parents=True, exist_ok=True)
        for case_dir in case_dirs:
            case_result = self.infer_case(case_dir)
            results.append(case_result)
            if per_case_output_dir is not None:
                case_id = str(case_result.get("case", case_dir.name))
                case_path = per_case_output_dir / f"case_{case_id}.json"
                case_path.write_text(json.dumps(case_result, ensure_ascii=False, indent=2), encoding="utf-8")
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
