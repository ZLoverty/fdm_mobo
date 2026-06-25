"""配置层：参数 / 目标 / Config(YAML 解析 + 字段顺序 + 指纹)。
torch 仅在 bounds_tensor() 内惰性导入，保证导入本模块不触发 torch。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Param:
    name: str
    low: float
    high: float


@dataclass
class Objective:
    name: str
    goal: str  # 'max' 越大越好 / 'min' 越小越好

    @property
    def sign(self) -> float:
        return 1.0 if self.goal == "max" else -1.0


_DEFAULTS = dict(n_init=6, seed=0, batch=1, num_restarts=12, raw_samples=256, mc_samples=128)


@dataclass(frozen=True)
class Config:
    params: list[Param]
    objectives: list[Objective]
    n_init: int = 6
    seed: int = 0
    batch: int = 1
    num_restarts: int = 12
    raw_samples: int = 256
    mc_samples: int = 128

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        params = [Param(p["name"], float(p["low"]), float(p["high"])) for p in d["params"]]
        objs = [Objective(o["name"], o["goal"]) for o in d["objectives"]]
        kw = {k: d.get(k, v) for k, v in _DEFAULTS.items()}
        return cls(params=params, objectives=objs, **kw)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def fieldnames(self) -> list[str]:
        return (["idx", "phase"]
                + [p.name for p in self.params]
                + [o.name for o in self.objectives]
                + ["time"])

    def fingerprint(self) -> str:
        payload = {
            "params": [[p.name, p.low, p.high] for p in self.params],
            "objectives": [[o.name, o.goal] for o in self.objectives],
            "n_init": self.n_init,
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def bounds_tensor(self):
        import torch
        return torch.tensor([[p.low for p in self.params],
                             [p.high for p in self.params]], dtype=torch.double)
