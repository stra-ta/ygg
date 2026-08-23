"""
Ygg model configuration.

Single source of truth for model + training hyperparameters. The dataclass is
serializable to / from plain dicts and JSON so runs can be reproduced exactly.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class ModelConfig:
    # --- Core transformer ---
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # --- Hierarchy ---
    local_layers: int = 6
    global_layers: int = 4
    window_size: int = 512
    # Number of windows per training sequence (seq_len = n_windows * window_size).
    # Drives the global transformer length and the temporal-consistency signal.
    n_windows: int = 4

    # --- Vocabulary / sequence ---
    vocab_size: int = 10000
    max_seq_len: int = 512
    # Input token width produced by dataset.events_to_tokens:
    # [event_type, thread, cpu, log_dt, arg0, arg1, arg2]
    token_dim: int = 7

    # --- Objective weights ---
    masked_weight: float = 1.0
    next_weight: float = 1.0
    contrastive_weight: float = 0.5
    temporal_weight: float = 0.3

    # --- Objective hyperparameters ---
    mask_ratio: float = 0.15
    mask_token_id: int = 0
    temperature: float = 0.07

    # --- Training ---
    batch_size: int = 32
    grad_accum: int = 4
    lr: float = 3e-4
    warmup: int = 1000
    epochs: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # --- Schedule / steps ---
    # Total optimizer steps used for the cosine decay tail. If <= 0 it is
    # derived from the dataset size at runtime.
    total_steps: int = 0

    # --- Logging / checkpointing ---
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 500
    use_wandb: bool = False
    run_name: str = "ygg-run"

    # --- Distributed ---
    # When device_count() > 1 and this is True, training is data-parallel via
    # jax.pmap ("Vanta" multi-GPU). Single device runs ignore it.
    distributed: bool = False

    # --- Embedding pool method for the execution repr ---
    pool: str = "mean"  # "mean" | "max" | "cls"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __str__(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
