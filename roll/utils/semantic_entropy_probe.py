from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


PROBE_SPECS = {
    "qwen3-4b": {
        "layer": 35,
        "hidden_size": 2560,
        "filename": "qwen3-4b_semantic_entropy_probe.pt",
    },
    "qwen3-8b": {
        "layer": 34,
        "hidden_size": 4096,
        "filename": "qwen3-8b_semantic_entropy_probe.pt",
    },
}


def semantic_entropy_top_enabled(pipeline_config) -> bool:
    return (
        getattr(pipeline_config, "entropy_top_ratio", None) is not None
        and getattr(pipeline_config, "entropy_top_score_type", "entropy") == "semantic_entropy"
    )


def infer_probe_key(model_name_or_path: Optional[str], requested: str = "auto") -> str:
    if requested and requested != "auto":
        key = requested.lower().replace("_", "-")
        if key in PROBE_SPECS:
            return key
        raise ValueError(f"Unsupported semantic entropy probe model: {requested}")

    model_name = str(model_name_or_path or "").lower().replace("_", "-")
    if "qwen3-4b" in model_name:
        return "qwen3-4b"
    if "qwen3-8b" in model_name:
        return "qwen3-8b"
    raise ValueError(
        "semantic_entropy_probe_model=auto only supports Qwen3-4B and Qwen3-8B. "
        f"Could not infer from model path: {model_name_or_path}"
    )


def resolve_probe_path(pipeline_config) -> Path:
    probe_path = getattr(pipeline_config, "semantic_entropy_probe_path", None)
    if probe_path:
        return Path(probe_path)

    key = infer_probe_key(
        getattr(pipeline_config, "pretrain", None),
        getattr(pipeline_config, "semantic_entropy_probe_model", "auto"),
    )
    probe_dir = Path(getattr(pipeline_config, "semantic_entropy_probe_dir", "./models/semantic_entropy_probes"))
    return probe_dir / PROBE_SPECS[key]["filename"]


def _to_int(value) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


@dataclass
class SemanticEntropyProbe:
    model_key: str
    layer: int
    hidden_size: int
    mean: torch.Tensor
    scale: torch.Tensor
    x_offset: torch.Tensor
    coef: torch.Tensor
    intercept: torch.Tensor
    path: Path

    @classmethod
    def from_pipeline_config(cls, pipeline_config, device: torch.device | str) -> "SemanticEntropyProbe":
        key = infer_probe_key(
            getattr(pipeline_config, "pretrain", None),
            getattr(pipeline_config, "semantic_entropy_probe_model", "auto"),
        )
        expected = PROBE_SPECS[key]
        path = resolve_probe_path(pipeline_config)
        if not path.exists():
            raise FileNotFoundError(
                f"Semantic entropy probe not found: {path}. "
                "Train it first and copy it under ROLL/models/semantic_entropy_probes/."
            )

        checkpoint = torch.load(path, map_location=device)
        layer = _to_int(checkpoint.get("layer", expected["layer"]))
        hidden_size = _to_int(checkpoint.get("hidden_size", expected["hidden_size"]))
        if layer != expected["layer"]:
            raise ValueError(f"{key} probe layer mismatch: expected {expected['layer']}, got {layer}")
        if hidden_size != expected["hidden_size"]:
            raise ValueError(f"{key} probe hidden_size mismatch: expected {expected['hidden_size']}, got {hidden_size}")

        coef = torch.as_tensor(checkpoint["coef"], dtype=torch.float32, device=device)
        if coef.ndim == 2:
            if coef.shape[0] == hidden_size and coef.shape[1] == 1:
                coef = coef[:, 0]
            elif coef.shape[0] == 1 and coef.shape[1] == hidden_size:
                coef = coef[0]
        coef = coef.reshape(hidden_size)

        return cls(
            model_key=key,
            layer=layer,
            hidden_size=hidden_size,
            mean=torch.as_tensor(checkpoint["mean"], dtype=torch.float32, device=device).reshape(hidden_size),
            scale=torch.as_tensor(checkpoint["scale"], dtype=torch.float32, device=device).reshape(hidden_size),
            x_offset=torch.as_tensor(
                checkpoint.get("x_offset", torch.zeros(hidden_size)),
                dtype=torch.float32,
                device=device,
            ).reshape(hidden_size),
            coef=coef,
            intercept=torch.as_tensor(checkpoint["intercept"], dtype=torch.float32, device=device).reshape(()),
            path=path,
        )

    def score_response_tokens(self, hidden_states: torch.Tensor, response_mask: torch.Tensor):
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states with shape (batch, seq, hidden), got {hidden_states.shape}")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Semantic entropy hidden size mismatch: expected {self.hidden_size}, got {hidden_states.size(-1)}"
            )
        if response_mask.size(1) != hidden_states.size(1):
            raise ValueError(
                f"response_mask and hidden_states sequence lengths differ: "
                f"{response_mask.size(1)} vs {hidden_states.size(1)}"
            )

        x = hidden_states.to(dtype=torch.float32)
        scale = torch.clamp(self.scale, min=1e-6)
        token_scores = (((x - self.mean) / scale) - self.x_offset).matmul(self.coef) + self.intercept

        # Score token x+1 with the hidden state at token x. The first response
        # token has no response-token predecessor, so it is excluded from top-k.
        scores = token_scores[:, :-1]
        valid_mask = response_mask[:, 1:].bool() & response_mask[:, :-1].bool()
        scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
        return scores, valid_mask
