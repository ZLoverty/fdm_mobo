# FDM MOBO 配置外提与实验文件夹 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把硬编码的控制参数与优化目标提取成每个实验独立的 YAML 配置，配置与数据同处一个实验文件夹，改维度即新建实验，并用指纹检测防止配置与旧数据不一致。

**Architecture:** 新增 torch-free 的 `fdm_config.py`（`Param`/`Objective`/`Config`：YAML 解析、字段顺序、指纹）。`fdm_mobo.py` 引入 `Experiment`（绑定一个 `experiments/<name>/` 文件夹：数据读写、冲突检测、Sobol 初始化、BO 建议），去掉模块全局 `PARAMS`/`OBJECTIVES`，CLI 各子命令加 `--exp`。`fdm_mobo_gui.py` 持有 `self.exp`，顶部加实验下拉框 + 新建按钮，切换时按当前配置重建表格列 / 待测面板 / 可视化控件。

**Tech Stack:** Python 3.12，pyyaml，BoTorch/torch，tkinter，matplotlib，pytest。

## Global Constraints

- Python ≥ 3.12（已确认 3.12.10）。
- 配置格式：YAML，依赖 `pyyaml`（已安装 6.0.3）。
- 指纹只覆盖"定义实验身份"的字段：`params`(name/low/high)、`objectives`(name/goal)、`n_init`、`seed`。acqf 预算（batch/num_restarts/raw_samples/mc_samples）改动**不**触发冲突。
- 改维度 = 新建实验文件夹，旧数据整套作废、不迁移补值。
- 冲突时**拒绝加载**该实验数据，绝不静默喂入模型。
- 冲突提示文案（逐字）：`配置与已采集数据不匹配，旧数据已失效。\n请改回配置，或用『新建实验』复制此配置再改维度。`
- 数值精度保持现状：BoTorch 用 `torch.set_default_dtype(torch.double)`；Sobol 点 `round(..., 3)`。
- **懒加载 torch/botorch**：`fdm_mobo.py` 顶部**不再** import torch/botorch/gpytorch；导入 `fdm_mobo` 模块本身不得触发 torch。torch 与 botorch 符号在需要它们的 `Experiment` 方法内部局部 import（`to_XY`/`fit_model`/`suggest_next`/`pareto_and_hv`/`init_sobol`）。每个创建张量的方法在开头执行 `import torch; torch.set_default_dtype(torch.double)`（重复调用安全）。目的：数据层/迁移/发现/CLI 解析/GUI 启动不依赖 torch。
- `experiments/` 为运行时数据，加入 `.gitignore`（但保留 spec/plan 文档）。

---

## File Structure

- **Create `fdm_config.py`** — `Param`, `Objective`, `Config`。纯 Python + yaml，torch 仅在 `bounds_tensor()` 内惰性导入，保证 import 该模块不触发 torch（单测快）。
- **Modify `fdm_mobo.py`** — 删除模块全局 `PARAMS`/`OBJECTIVES`/`N_INIT`/`FIELDNAMES` 等；新增 `Experiment` 类、实验发现/迁移辅助函数、`LEGACY_CONFIG_YAML` 常量；CLI 子命令加 `--exp`。
- **Modify `fdm_mobo_gui.py`** — 引入 `self.exp`，顶部实验选择栏，`_rebuild_for_config()`，`_load_experiment()`，新建实验流程；所有刷新/绘图改走 `self.exp` 与 `self.cfg`。
- **Create `tests/test_config.py`** — Config 解析/字段/指纹。
- **Create `tests/test_experiment.py`** — 数据读写、冲突检测、init_sobol、BO 冒烟。
- **Create `tests/test_migration.py`** — 旧根目录 trials.csv → experiments/default/。
- **Modify `.gitignore`** — 忽略 `experiments/`。

---

## Task 1: Config 模块（解析 + 字段顺序 + 指纹）

**Files:**
- Create: `fdm_config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无（基础任务）。
- Produces:
  - `@dataclass class Param: name: str; low: float; high: float`
  - `@dataclass class Objective: name: str; goal: str`（property `sign -> float`：max→1.0，min→-1.0）
  - `@dataclass(frozen=True) class Config`，字段：`params: list[Param]`, `objectives: list[Objective]`, `n_init: int=6`, `seed: int=0`, `batch: int=1`, `num_restarts: int=12`, `raw_samples: int=256`, `mc_samples: int=128`
    - `Config.from_yaml(path: str | Path) -> Config`
    - `Config.from_dict(d: dict) -> Config`
    - `cfg.fieldnames() -> list[str]` → `["idx","phase", *param_names, *obj_names, "time"]`
    - `cfg.fingerprint() -> str`（sha256 hex）
    - `cfg.bounds_tensor()`（惰性 import torch，返回 `[[lows],[highs]]` double 张量）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
from pathlib import Path
import fdm_config as fc


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


BASE = """\
params:
  - {name: fan,  low: 0,   high: 100}
  - {name: flow, low: 0.9, high: 1.1}
objectives:
  - {name: surface, goal: max}
  - {name: TS,      goal: min}
n_init: 6
seed: 0
"""


def test_from_yaml_parses_params_and_objectives(tmp_path):
    cfg = fc.Config.from_yaml(_write(tmp_path, BASE))
    assert [p.name for p in cfg.params] == ["fan", "flow"]
    assert cfg.params[1].low == 0.9 and cfg.params[1].high == 1.1
    assert [o.name for o in cfg.objectives] == ["surface", "TS"]
    assert cfg.objectives[0].sign == 1.0
    assert cfg.objectives[1].sign == -1.0


def test_defaults_applied_when_missing(tmp_path):
    text = """\
params:
  - {name: fan, low: 0, high: 100}
objectives:
  - {name: TS, goal: max}
"""
    cfg = fc.Config.from_yaml(_write(tmp_path, text))
    assert cfg.n_init == 6 and cfg.seed == 0 and cfg.batch == 1
    assert cfg.num_restarts == 12 and cfg.raw_samples == 256 and cfg.mc_samples == 128


def test_fieldnames_order(tmp_path):
    cfg = fc.Config.from_yaml(_write(tmp_path, BASE))
    assert cfg.fieldnames() == ["idx", "phase", "fan", "flow", "surface", "TS", "time"]


def test_fingerprint_stable_across_reload(tmp_path):
    cfg1 = fc.Config.from_yaml(_write(tmp_path, BASE))
    cfg2 = fc.Config.from_yaml(_write(tmp_path, BASE))
    assert cfg1.fingerprint() == cfg2.fingerprint()


def test_fingerprint_changes_on_added_dimension(tmp_path):
    cfg1 = fc.Config.from_yaml(_write(tmp_path, BASE))
    text2 = BASE + "  # add a dim below\n"  # comment alone must NOT change it
    cfg_comment = fc.Config.from_yaml(_write(tmp_path, text2))
    assert cfg1.fingerprint() == cfg_comment.fingerprint()
    text3 = BASE.replace(
        "  - {name: flow, low: 0.9, high: 1.1}\n",
        "  - {name: flow, low: 0.9, high: 1.1}\n  - {name: temp, low: 190, high: 220}\n",
    )
    cfg3 = fc.Config.from_yaml(_write(tmp_path, text3))
    assert cfg1.fingerprint() != cfg3.fingerprint()


def test_fingerprint_changes_on_range_change(tmp_path):
    cfg1 = fc.Config.from_yaml(_write(tmp_path, BASE))
    cfg2 = fc.Config.from_yaml(_write(tmp_path, BASE.replace("high: 100", "high: 80")))
    assert cfg1.fingerprint() != cfg2.fingerprint()


def test_fingerprint_ignores_acqf_budget(tmp_path):
    cfg1 = fc.Config.from_yaml(_write(tmp_path, BASE))
    cfg2 = fc.Config.from_yaml(_write(tmp_path, BASE + "num_restarts: 99\nmc_samples: 64\n"))
    assert cfg1.fingerprint() == cfg2.fingerprint()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd "c:/Users/zhengyang/Documents/GitHub/fdm_mobo" && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fdm_config'`

- [ ] **Step 3: 写最小实现**

```python
# fdm_config.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add fdm_config.py tests/test_config.py
git commit -m "feat: add Config module (yaml parse, fieldnames, fingerprint)"
```

---

## Task 2: Experiment 数据层（trials 读写 + meta + 冲突检测）

**Files:**
- Modify: `fdm_mobo.py`（在 import 区下、删除旧全局 `PARAMS`/`OBJECTIVES`/`N_INIT`/`SEED`/`BATCH`/`TRIALS_CSV`/`FIELDNAMES`/`Param`/`Objective`，改为 `from fdm_config import Param, Objective, Config`；新增 `Experiment` 类）
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `fdm_config.Config`（Task 1）。
- Produces:
  - `class Experiment(dir: str | Path, cfg: Config | None = None)`；属性 `dir: Path`、`cfg: Config`（cfg 为 None 时从 `dir/config.yaml` 加载）、`trials_path`、`meta_path`。
  - `exp.load_trials() -> list[dict]`
  - `exp.save_trials(rows: list[dict]) -> None`
  - `exp.is_complete(r: dict) -> bool`
  - `exp.next_idx(rows) -> int`
  - `exp.write_meta() -> None`（写 `{config_fingerprint, created}`）
  - `exp.check_conflict() -> str | None`（无 meta 返回 None；指纹不符返回提示文案）
  - 模块常量 `CONFLICT_MSG: str`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_experiment.py
import math
from pathlib import Path

import fdm_config as fc
import fdm_mobo as core

BASE_YAML = """\
params:
  - {name: fan,  low: 0,   high: 100}
  - {name: flow, low: 0.9, high: 1.1}
objectives:
  - {name: surface, goal: max}
  - {name: TS,      goal: max}
n_init: 6
seed: 0
"""


def _make_exp(tmp_path, yaml_text=BASE_YAML):
    d = tmp_path / "exp1"
    d.mkdir()
    (d / "config.yaml").write_text(yaml_text, encoding="utf-8")
    return core.Experiment(d)


def test_save_load_roundtrip_with_nan(tmp_path):
    exp = _make_exp(tmp_path)
    rows = [
        {"idx": 0, "phase": "init", "fan": 20.0, "flow": 1.0,
         "surface": math.nan, "TS": math.nan, "time": ""},
        {"idx": 1, "phase": "init", "fan": 50.0, "flow": 1.0,
         "surface": 7.0, "TS": 30.0, "time": "2026-06-25T00:00:00"},
    ]
    exp.save_trials(rows)
    back = exp.load_trials()
    assert back[0]["idx"] == 0 and back[0]["fan"] == 20.0
    assert math.isnan(back[0]["surface"])
    assert back[1]["surface"] == 7.0 and back[1]["TS"] == 30.0
    assert exp.is_complete(back[1]) and not exp.is_complete(back[0])


def test_next_idx(tmp_path):
    exp = _make_exp(tmp_path)
    assert exp.next_idx([]) == 0
    assert exp.next_idx([{"idx": 0}, {"idx": 3}]) == 4


def test_check_conflict_none_without_meta(tmp_path):
    exp = _make_exp(tmp_path)
    assert exp.check_conflict() is None


def test_check_conflict_ok_after_write_meta(tmp_path):
    exp = _make_exp(tmp_path)
    exp.write_meta()
    assert exp.check_conflict() is None


def test_check_conflict_detects_changed_config(tmp_path):
    exp = _make_exp(tmp_path)
    exp.write_meta()
    # 用户事后改了维度范围
    (exp.dir / "config.yaml").write_text(
        BASE_YAML.replace("high: 100", "high: 80"), encoding="utf-8")
    exp2 = core.Experiment(exp.dir)  # 重新加载新配置
    msg = exp2.check_conflict()
    assert msg is not None and "旧数据已失效" in msg
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: FAIL — `AttributeError: module 'fdm_mobo' has no attribute 'Experiment'`（或 import 期因旧全局被删而报错——按 Step 3 一并清理）

- [ ] **Step 3: 写实现**

**先改顶部 import 区（懒加载 torch）**：删除 `fdm_mobo.py` 顶部所有 torch/botorch/gpytorch 的 import（`import torch`、`from botorch...`、`from gpytorch...`）以及模块级 `torch.set_default_dtype(torch.double)` 那一行。顶部只保留 stdlib：`argparse, csv, math, os`（原有）+ 新增 `import json` 与 `from pathlib import Path`。torch/botorch 将在 Task 3 的 BO 方法内部局部 import。导入 `fdm_mobo` 模块本身不得触发 torch（Task 2 的测试 `import fdm_mobo as core` 必须在无 torch 时也能成功）。

然后把原配置区（`@dataclass Param/Objective`、`PARAMS`、`OBJECTIVES`、`N_INIT`、`BATCH`、`TRIALS_CSV`、`SEED`、`NUM_RESTARTS`、`RAW_SAMPLES`、`MC_SAMPLES`、`FIELDNAMES`）整段删除，替换为：

```python
from pathlib import Path
import json

from fdm_config import Param, Objective, Config  # noqa: F401  (Param/Objective re-export)

CONFLICT_MSG = (
    "配置与已采集数据不匹配，旧数据已失效。\n"
    "请改回配置，或用『新建实验』复制此配置再改维度。"
)


class Experiment:
    """绑定一个 experiments/<name>/ 文件夹：配置 + trials.csv + meta.json。"""

    def __init__(self, dir, cfg: Config | None = None):
        self.dir = Path(dir)
        self.cfg = cfg if cfg is not None else Config.from_yaml(self.dir / "config.yaml")

    # ---- 路径 ----
    @property
    def trials_path(self) -> Path:
        return self.dir / "trials.csv"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    # ---- 数据读写 ----
    def load_trials(self) -> list[dict]:
        path = self.trials_path
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["idx"] = int(r["idx"])
            for p in self.cfg.params:
                r[p.name] = float(r[p.name])
            for o in self.cfg.objectives:
                v = r.get(o.name, "")
                r[o.name] = float(v) if v not in ("", None) else math.nan
        return rows

    def save_trials(self, rows: list[dict]) -> None:
        fields = self.cfg.fieldnames()
        with open(self.trials_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                out = {k: r.get(k, "") for k in fields}
                for o in self.cfg.objectives:
                    v = r.get(o.name, math.nan)
                    out[o.name] = "" if (isinstance(v, float) and math.isnan(v)) else v
                w.writerow(out)

    def is_complete(self, r: dict) -> bool:
        return all(not math.isnan(r[o.name]) for o in self.cfg.objectives)

    def next_idx(self, rows: list[dict]) -> int:
        return (max((r["idx"] for r in rows), default=-1)) + 1

    # ---- meta / 冲突检测 ----
    def write_meta(self) -> None:
        meta = {
            "config_fingerprint": self.cfg.fingerprint(),
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_meta(self) -> dict | None:
        if not self.meta_path.exists():
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def check_conflict(self) -> str | None:
        meta = self._read_meta()
        if meta is None:
            return None
        if meta.get("config_fingerprint") != self.cfg.fingerprint():
            return CONFLICT_MSG
        return None
```

注意：`csv`、`math`、`datetime` 已在原文件 import；确认 `import json`、`from pathlib import Path` 已加。删除旧的模块级 `load_trials/save_trials/is_complete/next_idx` 自由函数（它们将被 Task 3 的 Experiment 方法取代；本步骤先删 `load_trials/save_trials/is_complete/next_idx` 四个自由函数，避免与方法重名混淆）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add fdm_mobo.py tests/test_experiment.py
git commit -m "feat: add Experiment data layer with conflict detection"
```

---

## Task 3: Experiment 的 Sobol 初始化与 BO 建议

**Files:**
- Modify: `fdm_mobo.py`（把原自由函数 `bounds_tensor/to_XY/fit_model/ref_point/suggest_next/pareto_and_hv` 改写为 `Experiment` 方法或接受 `cfg` 的私有 helper；新增 `init_sobol`）
- Test: `tests/test_experiment.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `Experiment`、`fdm_config.Config.bounds_tensor`。
- Produces:
  - `exp.init_sobol() -> None`（若已有 trials 抛 `RuntimeError`；否则生成 `cfg.n_init` 个 Sobol 点写入 trials.csv，并 `write_meta()`）
  - `exp.to_XY()` → `(X, Y, done)`（Y 已乘 sign，内部最大化）
  - `exp.fit_model(X, Y)`
  - `exp.suggest_next() -> torch.Tensor`（已完成 < 2 抛 `RuntimeError`）
  - `exp.pareto_and_hv()` → `(pareto_rows, hv_float)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_experiment.py （追加到文件末尾）
def test_init_sobol_creates_points_and_meta(tmp_path):
    exp = _make_exp(tmp_path)
    exp.init_sobol()
    rows = exp.load_trials()
    assert len(rows) == exp.cfg.n_init
    assert all(not exp.is_complete(r) for r in rows)        # 测量值留空
    for r in rows:
        assert 0.0 <= r["fan"] <= 100.0 and 0.9 <= r["flow"] <= 1.1
    assert exp.meta_path.exists()
    assert exp.check_conflict() is None                      # init 后指纹一致


def test_init_sobol_refuses_when_data_exists(tmp_path):
    import pytest
    exp = _make_exp(tmp_path)
    exp.save_trials([{"idx": 0, "phase": "init", "fan": 1.0, "flow": 1.0,
                      "surface": math.nan, "TS": math.nan, "time": ""}])
    with pytest.raises(RuntimeError):
        exp.init_sobol()


def test_suggest_next_smoke(tmp_path):
    exp = _make_exp(tmp_path)
    rows = [
        {"idx": 0, "phase": "init", "fan": 20.0, "flow": 0.95, "surface": 6.0, "TS": 28.0, "time": "t"},
        {"idx": 1, "phase": "init", "fan": 80.0, "flow": 1.05, "surface": 8.0, "TS": 22.0, "time": "t"},
        {"idx": 2, "phase": "init", "fan": 50.0, "flow": 1.00, "surface": 7.0, "TS": 30.0, "time": "t"},
    ]
    exp.save_trials(rows)
    cand = exp.suggest_next()
    assert cand.shape == (exp.cfg.batch, len(exp.cfg.params))
    c = cand[0].tolist()
    assert 0.0 <= c[0] <= 100.0 and 0.9 <= c[1] <= 1.1


def test_pareto_and_hv_smoke(tmp_path):
    exp = _make_exp(tmp_path)
    exp.save_trials([
        {"idx": 0, "phase": "init", "fan": 20.0, "flow": 0.95, "surface": 6.0, "TS": 28.0, "time": "t"},
        {"idx": 1, "phase": "init", "fan": 80.0, "flow": 1.05, "surface": 8.0, "TS": 22.0, "time": "t"},
    ])
    pareto, hv = exp.pareto_and_hv()
    assert len(pareto) >= 1 and isinstance(hv, float)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_experiment.py -k "sobol or suggest or pareto" -v`
Expected: FAIL — `AttributeError: 'Experiment' object has no attribute 'init_sobol'`

- [ ] **Step 3: 写实现**

在 `Experiment` 类中追加以下方法（把原模块自由函数 `bounds_tensor/to_XY/fit_model/ref_point/suggest_next/pareto_and_hv` 删除，逻辑迁入这里，参数从全局改为 `self.cfg`）。

**懒加载约定**（见 Global Constraints）：torch/botorch 符号在方法内部局部 import，顶部不再有这些 import。每个创建张量的方法开头执行 `import torch; torch.set_default_dtype(torch.double)`（重复调用安全）。具体如下：

```python
    # ---- 张量 / 模型 ----
    def to_XY(self):
        import torch
        torch.set_default_dtype(torch.double)
        done = self.load_trials_done()
        X = torch.tensor([[r[p.name] for p in self.cfg.params] for r in done])
        Y_raw = torch.tensor([[r[o.name] for o in self.cfg.objectives] for r in done])
        signs = torch.tensor([o.sign for o in self.cfg.objectives])
        return X, Y_raw * signs, done

    def load_trials_done(self) -> list[dict]:
        return [r for r in self.load_trials() if self.is_complete(r)]

    def fit_model(self, X, Y):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        model = SingleTaskGP(
            X, Y,
            input_transform=Normalize(d=X.shape[-1], bounds=self.cfg.bounds_tensor()),
            outcome_transform=Standardize(m=Y.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    @staticmethod
    def _ref_point(Y):
        import torch
        mn = Y.min(dim=0).values
        rng = Y.max(dim=0).values - mn
        rng = torch.where(rng > 0, rng, torch.ones_like(rng))
        return mn - 0.1 * rng

    def suggest_next(self):
        import torch
        torch.set_default_dtype(torch.double)
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.optim import optimize_acqf
        X, Y, done = self.to_XY()
        if len(done) < 2:
            raise RuntimeError("已完成的实验少于 2 个，先回填更多点再 suggest。")
        model = self.fit_model(X, Y)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self._ref_point(Y).tolist(),
            X_baseline=X,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([self.cfg.mc_samples])),
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.cfg.bounds_tensor(),
            q=self.cfg.batch,
            num_restarts=self.cfg.num_restarts,
            raw_samples=self.cfg.raw_samples,
        )
        return candidates.detach()

    def pareto_and_hv(self):
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.box_decompositions.dominated import (
            DominatedPartitioning,
        )
        X, Y, done = self.to_XY()
        if len(done) == 0:
            return [], float("nan")
        mask = is_non_dominated(Y)
        hv = DominatedPartitioning(ref_point=self._ref_point(Y), Y=Y).compute_hypervolume().item()
        pareto = [done[i] for i in range(len(done)) if bool(mask[i])]
        return pareto, hv

    # ---- 初始化 ----
    def init_sobol(self) -> None:
        import torch
        torch.set_default_dtype(torch.double)
        from botorch.utils.sampling import draw_sobol_samples
        if self.load_trials():
            raise RuntimeError("该实验已有数据，init 会覆盖；请新建实验或先清空。")
        pts = draw_sobol_samples(
            bounds=self.cfg.bounds_tensor(), n=self.cfg.n_init, q=1, seed=self.cfg.seed
        ).squeeze(1)
        rows = []
        for i, pt in enumerate(pts):
            r = {"idx": i, "phase": "init", "time": ""}
            for j, par in enumerate(self.cfg.params):
                r[par.name] = round(float(pt[j]), 3)
            for o in self.cfg.objectives:
                r[o.name] = math.nan
            rows.append(r)
        self.save_trials(rows)
        self.write_meta()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: PASS（9 passed；BO 冒烟测试可能较慢，属正常）

- [ ] **Step 5: 提交**

```bash
git add fdm_mobo.py tests/test_experiment.py
git commit -m "feat: move Sobol init and BO suggest into Experiment"
```

---

## Task 4: 实验发现、当前指针与旧数据迁移

**Files:**
- Modify: `fdm_mobo.py`（新增实验发现/迁移辅助函数与 `LEGACY_CONFIG_YAML` 常量）
- Modify: `.gitignore`（忽略 `experiments/`）
- Test: `tests/test_migration.py`

**Interfaces:**
- Consumes: Task 1–3。
- Produces（均以 `root: Path` 为实验根，默认 `EXPERIMENTS_DIR = Path("experiments")`）：
  - `EXPERIMENTS_DIR: Path`
  - `LEGACY_CONFIG_YAML: str`（fan[0,100]/flow[0.9,1.1]，surface/TS 均 max，n_init=6 等）
  - `list_experiments(root=EXPERIMENTS_DIR) -> list[str]`（含 config.yaml 的子目录名，排序）
  - `get_current(root=EXPERIMENTS_DIR) -> str | None`（读 `<root>/.current`）
  - `set_current(name, root=EXPERIMENTS_DIR) -> None`
  - `create_experiment(name, template_yaml: str, root=EXPERIMENTS_DIR) -> Experiment`（建目录 + 写 config.yaml，不 init）
  - `migrate_legacy(legacy_csv="trials.csv", root=EXPERIMENTS_DIR) -> str | None`（若 root 不存在但 legacy_csv 存在：建 `default/`，写 `LEGACY_CONFIG_YAML`，移动 csv，写 meta，设 .current，返回 "default"；否则返回 None）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_migration.py
from pathlib import Path
import fdm_mobo as core


def test_migrate_legacy_creates_default(tmp_path):
    legacy = tmp_path / "trials.csv"
    legacy.write_text(
        "idx,phase,fan,flow,surface,TS,time\n"
        "0,init,20,1,,,\n"
        "1,init,50,1,7,30,2026-06-25T00:00:00\n",
        encoding="utf-8",
    )
    root = tmp_path / "experiments"
    name = core.migrate_legacy(legacy_csv=legacy, root=root)
    assert name == "default"
    d = root / "default"
    assert (d / "config.yaml").exists()
    assert (d / "trials.csv").exists()
    assert (d / "meta.json").exists()
    assert not legacy.exists()                       # 已移动
    assert core.get_current(root) == "default"
    exp = core.Experiment(d)
    assert exp.check_conflict() is None              # meta 指纹与生成 config 一致
    rows = exp.load_trials()
    assert len(rows) == 2 and rows[1]["surface"] == 7.0


def test_migrate_noop_when_root_exists(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()
    legacy = tmp_path / "trials.csv"
    legacy.write_text("idx,phase,fan,flow,surface,TS,time\n", encoding="utf-8")
    assert core.migrate_legacy(legacy_csv=legacy, root=root) is None


def test_list_and_current(tmp_path):
    root = tmp_path / "experiments"
    core.create_experiment("a", core.LEGACY_CONFIG_YAML, root=root)
    core.create_experiment("b", core.LEGACY_CONFIG_YAML, root=root)
    assert core.list_experiments(root) == ["a", "b"]
    core.set_current("b", root=root)
    assert core.get_current(root) == "b"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_migration.py -v`
Expected: FAIL — `AttributeError: module 'fdm_mobo' has no attribute 'migrate_legacy'`

- [ ] **Step 3: 写实现**

在 `fdm_mobo.py` 加入：

```python
import shutil

EXPERIMENTS_DIR = Path("experiments")

LEGACY_CONFIG_YAML = """\
# 由旧版硬编码配置迁移生成
params:
  - {name: fan,  low: 0.0,  high: 100.0}
  - {name: flow, low: 0.9,  high: 1.1}
objectives:
  - {name: surface, goal: max}
  - {name: TS,      goal: max}
n_init: 6
seed: 0
batch: 1
num_restarts: 12
raw_samples: 256
mc_samples: 128
"""


def list_experiments(root=EXPERIMENTS_DIR) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "config.yaml").exists())


def get_current(root=EXPERIMENTS_DIR) -> str | None:
    f = Path(root) / ".current"
    if not f.exists():
        return None
    name = f.read_text(encoding="utf-8").strip()
    return name or None


def set_current(name: str, root=EXPERIMENTS_DIR) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".current").write_text(name, encoding="utf-8")


def create_experiment(name: str, template_yaml: str, root=EXPERIMENTS_DIR) -> Experiment:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=False)
    (d / "config.yaml").write_text(template_yaml, encoding="utf-8")
    return Experiment(d)


def migrate_legacy(legacy_csv="trials.csv", root=EXPERIMENTS_DIR) -> str | None:
    root = Path(root)
    legacy_csv = Path(legacy_csv)
    if root.exists() or not legacy_csv.exists():
        return None
    exp = create_experiment("default", LEGACY_CONFIG_YAML, root=root)
    shutil.move(str(legacy_csv), str(exp.trials_path))
    exp.write_meta()
    set_current("default", root=root)
    return "default"
```

并在 `.gitignore` 末尾追加一行：

```
experiments/
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_migration.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add fdm_mobo.py tests/test_migration.py .gitignore
git commit -m "feat: add experiment discovery, current pointer, legacy migration"
```

---

## Task 5: CLI 子命令接 Experiment + `--exp`

**Files:**
- Modify: `fdm_mobo.py`（`cmd_init/cmd_suggest/cmd_record/cmd_status/cmd_apply`、`main`；删除旧的 `print_rows/fmt_params/fmt_objs` 对全局的依赖，改为接 `exp`）
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1–4。
- Produces:
  - `resolve_experiment(name: str | None, root=EXPERIMENTS_DIR) -> Experiment`（name 为空时用 `get_current()`，再退 `"default"`；目录不存在则报错退出）
  - 每个子命令解析 `--exp`，经 `resolve_experiment` 取 `exp` 后操作。
  - `main()` 在解析前调用 `migrate_legacy()` 自动迁移。

- [ ] **Step 1: 写失败测试**（用 subprocess 跑 init + status 冒烟）

```python
# tests/test_cli.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = """\
params:
  - {name: fan,  low: 0,   high: 100}
  - {name: flow, low: 0.9, high: 1.1}
objectives:
  - {name: surface, goal: max}
  - {name: TS,      goal: max}
n_init: 4
seed: 0
"""


def _run(args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "fdm_mobo.py"), *args],
                          cwd=cwd, capture_output=True, text=True)


def test_cli_init_then_status(tmp_path):
    exp_dir = tmp_path / "experiments" / "default"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "experiments" / ".current").write_text("default", encoding="utf-8")

    r1 = _run(["init"], cwd=tmp_path)
    assert r1.returncode == 0, r1.stderr
    assert (exp_dir / "trials.csv").exists()
    assert (exp_dir / "meta.json").exists()

    r2 = _run(["status"], cwd=tmp_path)
    assert r2.returncode == 0, r2.stderr
    assert "已完成 0" in r2.stdout
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL（CLI 仍引用已删除的全局，returncode != 0）

- [ ] **Step 3: 写实现**

替换 `fdm_mobo.py` 的展示/子命令/入口段。展示函数改为接 `exp`：

```python
def fmt_params(exp, r):
    return "  ".join(f"{p.name}={r[p.name]:g}" for p in exp.cfg.params)


def fmt_objs(exp, r):
    return "  ".join(
        f"{o.name}=" + ("?" if math.isnan(r[o.name]) else f"{r[o.name]:g}")
        for o in exp.cfg.objectives
    )


def print_rows(exp, rows):
    for r in rows:
        tag = "[待测]" if not exp.is_complete(r) else "      "
        print(f"  #{r['idx']:<3} {tag} {fmt_params(exp, r):<28} {fmt_objs(exp, r)}")


def resolve_experiment(name, root=EXPERIMENTS_DIR) -> Experiment:
    name = name or get_current(root) or "default"
    d = Path(root) / name
    if not (d / "config.yaml").exists():
        raise SystemExit(f"实验 '{name}' 不存在（{d}/config.yaml 未找到）。")
    exp = Experiment(d)
    conflict = exp.check_conflict()
    if conflict:
        raise SystemExit(conflict)
    return exp


def cmd_init(args):
    exp = resolve_experiment(args.exp)
    if exp.load_trials():
        print("该实验已有 trials.csv，跳过 init（用 suggest / record 继续）。")
        return
    exp.init_sobol()
    rows = exp.load_trials()
    print(f"已生成 {exp.cfg.n_init} 个初始点 -> {exp.trials_path}")
    print_rows(exp, rows)


def cmd_suggest(args):
    exp = resolve_experiment(args.exp)
    rows = exp.load_trials()
    if not rows:
        raise SystemExit("还没有 trials.csv，先运行 init。")
    pending = [r for r in rows if not exp.is_complete(r)]
    if pending:
        print("还有未测量的点，先 record 再 suggest：")
        print_rows(exp, pending)
        return
    cand = exp.suggest_next()
    start = exp.next_idx(rows)
    new = []
    for k, c in enumerate(cand):
        r = {"idx": start + k, "phase": "bo", "time": ""}
        for j, par in enumerate(exp.cfg.params):
            r[par.name] = round(float(c[j]), 3)
        for o in exp.cfg.objectives:
            r[o.name] = math.nan
        rows.append(r)
        new.append(r)
    exp.save_trials(rows)
    print("下一个实验点（qLogNEHVI）：")
    print_rows(exp, new)


def cmd_record(args):
    exp = resolve_experiment(args.exp)
    rows = exp.load_trials()
    by_idx = {r["idx"]: r for r in rows}
    if args.idx is not None:
        r = by_idx.get(args.idx)
        if r is None:
            raise SystemExit(f"找不到 #{args.idx}")
        if args.values:
            if len(args.values) != len(exp.cfg.objectives):
                raise SystemExit(f"需要 {len(exp.cfg.objectives)} 个值: {[o.name for o in exp.cfg.objectives]}")
            for o, v in zip(exp.cfg.objectives, args.values):
                r[o.name] = float(v)
        else:
            for o in exp.cfg.objectives:
                v = input(f"#{args.idx} {o.name} = ").strip()
                if v:
                    r[o.name] = float(v)
        r["time"] = datetime.now().isoformat(timespec="seconds")
        exp.save_trials(rows)
        print("已保存 #%d: %s" % (args.idx, fmt_objs(exp, r)))
        return
    pending = [r for r in rows if not exp.is_complete(r)]
    if not pending:
        print("没有待测量的点。")
        return
    for r in pending:
        print(f"\n#{r['idx']}  {fmt_params(exp, r)}  (留空跳过)")
        for o in exp.cfg.objectives:
            v = input(f"  {o.name} = ").strip()
            if v:
                r[o.name] = float(v)
        if exp.is_complete(r):
            r["time"] = datetime.now().isoformat(timespec="seconds")
    exp.save_trials(rows)
    print("\n已保存。")


def cmd_status(args):
    exp = resolve_experiment(args.exp)
    rows = exp.load_trials()
    if not rows:
        print("还没有数据。先 init。")
        return
    done = [r for r in rows if exp.is_complete(r)]
    print(f"实验总数 {len(rows)}，已完成 {len(done)}，待测 {len(rows) - len(done)}")
    print_rows(exp, rows)
    if done:
        pareto, hv = exp.pareto_and_hv()
        print(f"\n当前 Pareto 前沿（{len(pareto)} 个非支配点）：")
        print_rows(exp, pareto)
        print(f"\n超体积 hypervolume = {hv:.4g}")


def cmd_apply(args):
    exp = resolve_experiment(args.exp)
    r = {x["idx"]: x for x in exp.load_trials()}.get(args.idx)
    if r is None:
        raise SystemExit(f"找不到 #{args.idx}")
    fan = r["fan"]; flow = r["flow"]
    fan_s = int(round(fan / 100.0 * 255))
    gcode = f"M106 S{fan_s}\nM221 S{int(round(flow))}"
    print(f"#{args.idx} -> fan {fan:g}% (M106 S{fan_s}), flow {flow:g}% (M221 S{int(round(flow))})")
    if not args.host:
        print("(未提供 --host，仅打印 gcode，不下发)")
        print(gcode)
        return
    import json as _json
    import urllib.request
    url = args.host.rstrip("/") + "/printer/gcode/script"
    req = urllib.request.Request(
        url, data=_json.dumps({"script": gcode}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("Moonraker:", resp.status, resp.read().decode()[:200])


def main():
    migrate_legacy()
    ap = argparse.ArgumentParser(description="FDM 多目标贝叶斯优化")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_exp(p):
        p.add_argument("--exp", type=str, default=None, help="实验名（默认读 experiments/.current）")

    p_init = sub.add_parser("init", help="生成初始 Sobol 实验点"); add_exp(p_init); p_init.set_defaults(func=cmd_init)
    p_sug = sub.add_parser("suggest", help="给出下一个实验点"); add_exp(p_sug); p_sug.set_defaults(func=cmd_suggest)
    p_st = sub.add_parser("status", help="查看实验 / Pareto / 超体积"); add_exp(p_st); p_st.set_defaults(func=cmd_status)

    pr = sub.add_parser("record", help="回填测量值"); add_exp(pr)
    pr.add_argument("--idx", type=int)
    pr.add_argument("values", nargs="*")
    pr.set_defaults(func=cmd_record)

    pa = sub.add_parser("apply", help="(可选) 经 Moonraker 下发 fan/flow"); add_exp(pa)
    pa.add_argument("--idx", type=int, required=True)
    pa.add_argument("--host", type=str, default="")
    pa.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)
```

删除旧的模块级 `fmt_params/fmt_objs/print_rows`（无 exp 参数版）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cli.py -v && python -m pytest -q`
Expected: PASS（全部测试通过）

- [ ] **Step 5: 提交**

```bash
git add fdm_mobo.py tests/test_cli.py
git commit -m "feat: route CLI subcommands through Experiment with --exp"
```

---

## Task 6: GUI 接 Experiment + 实验下拉/新建 + 按配置重建界面

**Files:**
- Modify: `fdm_mobo_gui.py`
- 验证：手动启动 GUI（GUI 不做自动化单测，提供导入冒烟 + 手动验证清单）

**Interfaces:**
- Consumes: Task 1–5（`core.Experiment`、`core.list_experiments/get_current/set_current/create_experiment/migrate_legacy`、`core.EXPERIMENTS_DIR`）。
- Produces: GUI 持有 `self.exp: Experiment` 与 `self.cfg = self.exp.cfg`；新增 `_build_exp_bar`、`_rebuild_for_config`、`_load_experiment`、`_new_experiment`。

- [ ] **Step 1: 导入冒烟测试（确保重构后模块可导入、可构造配置）**

```python
# tests/test_gui_smoke.py
import importlib


def test_gui_module_imports():
    # 仅验证模块语法与符号存在，不实例化 Tk（无显示环境）
    mod = importlib.import_module("fdm_mobo_gui")
    assert hasattr(mod, "App")
```

- [ ] **Step 2: 运行确认当前状态**

Run: `python -m pytest tests/test_gui_smoke.py -v`
Expected: 现状可能 PASS（旧 GUI 仍引用 core.PARAMS，import 期不报错）。记录基线，重构后须仍 PASS。

- [ ] **Step 3: 重构 GUI**

关键改动（按区块替换，保持其余绘图逻辑不变，仅把 `core.PARAMS`→`self.cfg.params`、`core.OBJECTIVES`→`self.cfg.objectives`、`core.load_trials()`→`self.exp.load_trials()`、`core.save_trials(rows)`→`self.exp.save_trials(rows)`、`core.is_complete(r)`→`self.exp.is_complete(r)`、`core.pareto_and_hv(rows)`→`self.exp.pareto_and_hv()`、`core.next_idx(rows)`→`self.exp.next_idx(rows)`、`core.suggest_next(rows)`→`self.exp.suggest_next()`）：

`__init__` 改为先解析当前实验：

```python
    def __init__(self):
        super().__init__()
        self.title("FDM 多目标贝叶斯优化")
        self.geometry("1300x840")
        self.minsize(960, 640)
        self._pareto_idxs: set[int] = set()
        self.rows: list[dict] = []
        self._cbar = None
        core.migrate_legacy()
        self._ensure_some_experiment()
        self.exp = core.Experiment(core.EXPERIMENTS_DIR / self._current_name())
        self.cfg = self.exp.cfg
        self._build_ui()
        self.after(80, self.refresh)

    def _current_name(self) -> str:
        return core.get_current() or (core.list_experiments() or ["default"])[0]

    def _ensure_some_experiment(self):
        if not core.list_experiments():
            core.create_experiment("default", core.LEGACY_CONFIG_YAML)
            core.set_current("default")
```

`_build_ui` 顶部加实验栏，并把"依赖配置的子部件"放进可重建容器：

```python
    def _build_ui(self) -> None:
        self._build_exp_bar(self)               # 顶部实验选择
        vpane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        vpane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        hpane = ttk.PanedWindow(vpane, orient=tk.HORIZONTAL)
        vpane.add(hpane, weight=2)
        left = ttk.Frame(hpane, padding=4); hpane.add(left, weight=3)
        self._left = left
        self._build_table(left)
        right = ttk.Frame(hpane, padding=4); hpane.add(right, weight=2)
        self._right = right
        self._build_current(right)
        self._build_stats(right)
        viz_outer = ttk.LabelFrame(vpane, text=" 可视化 ", padding=6)
        vpane.add(viz_outer, weight=3)
        self._viz_outer = viz_outer
        self._build_viz(viz_outer)

    def _build_exp_bar(self, parent) -> None:
        bar = ttk.Frame(parent, padding=(6, 4))
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="实验:", font=("", 10, "bold")).pack(side=tk.LEFT)
        self._expvar = tk.StringVar(value=self._current_name())
        self._expcombo = ttk.Combobox(bar, textvariable=self._expvar, state="readonly",
                                      values=core.list_experiments(), width=28)
        self._expcombo.pack(side=tk.LEFT, padx=6)
        self._expcombo.bind("<<ComboboxSelected>>",
                            lambda _e: self._load_experiment(self._expvar.get()))
        ttk.Button(bar, text="新建实验", command=self._new_experiment, width=10).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="打开配置", command=self._open_config, width=10).pack(side=tk.LEFT, padx=3)
```

切换 / 新建 / 打开配置：

```python
    def _load_experiment(self, name: str) -> None:
        exp = core.Experiment(core.EXPERIMENTS_DIR / name)
        conflict = exp.check_conflict()
        if conflict:
            messagebox.showerror("配置冲突", conflict)
            self._expvar.set(self.exp.dir.name)      # 回退到原选择
            return
        core.set_current(name)
        self.exp = exp
        self.cfg = exp.cfg
        self._rebuild_for_config()
        self.refresh()

    def _rebuild_for_config(self) -> None:
        for w in self._left.winfo_children():
            w.destroy()
        for w in self._right.winfo_children():
            w.destroy()
        for w in self._viz_outer.winfo_children():
            w.destroy()
        self._cbar = None
        self._build_table(self._left)
        self._build_current(self._right)
        self._build_stats(self._right)
        self._build_viz(self._viz_outer)

    def _new_experiment(self) -> None:
        name = simpledialog.askstring("新建实验", "实验名（将作为文件夹名）:", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in core.list_experiments():
            messagebox.showerror("已存在", f"实验 '{name}' 已存在")
            return
        template = (self.exp.dir / "config.yaml").read_text(encoding="utf-8")
        core.create_experiment(name, template)
        self._expcombo.config(values=core.list_experiments())
        self._expvar.set(name)
        self._load_experiment(name)
        try:
            os.startfile(str(core.EXPERIMENTS_DIR / name / "config.yaml"))  # noqa
        except Exception:
            pass
        messagebox.showinfo("新建实验",
            "已复制当前配置为模板并打开。\n编辑好维度后回来点『初始化』生成起始点。")

    def _open_config(self) -> None:
        try:
            os.startfile(str(self.exp.dir / "config.yaml"))  # noqa
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
```

`_init`（初始化按钮）改为走 `self.exp.init_sobol()` 并重新加载配置（用户可能刚改了 yaml）：

```python
    def _init(self) -> None:
        if self.rows:
            if not messagebox.askyesno("确认清除",
                    "将删除该实验所有数据并按当前 config.yaml 重新生成 Sobol 起始点，继续？"):
                return
        self.exp = core.Experiment(self.exp.dir)   # 重新读最新 config.yaml
        self.cfg = self.exp.cfg
        if self.exp.trials_path.exists():
            self.exp.trials_path.unlink()
        self.exp.init_sobol()
        self._rebuild_for_config()
        self.refresh()
```

`refresh`、`_refresh_table`、`_refresh_current`、`_refresh_stats`、`update_viz`、`_submit`、`_suggest`、`_on_dblclick` 中所有 `core.PARAMS/OBJECTIVES/load_trials/save_trials/is_complete/pareto_and_hv/next_idx/suggest_next` 按上面映射逐一替换为 `self.cfg.*` 与 `self.exp.*`。`_suggest` 内 `core.suggest_next(rows)` → `self.exp.suggest_next()`，`core.next_idx(rows)` → `self.exp.next_idx(rows)`，`core.save_trials(rows)` → `self.exp.save_trials(rows)`。

- [ ] **Step 4: 运行冒烟测试 + 手动验证**

Run: `python -m pytest tests/test_gui_smoke.py -v && python -m pytest -q`
Expected: PASS。

手动验证清单（启动 `python fdm_mobo_gui.py`）：
1. 首次启动：根目录旧 `trials.csv` 被迁移到 `experiments/default/`，界面显示原 6 个点。
2. 下拉框显示 `default`；统计/表格/图正常。
3. 「新建实验」→ 输入 `3parm_test` → 弹出系统编辑器打开 config.yaml；在 params 下加一行 `- {name: temp, low: 190, high: 220}` 保存。
4. 回到 GUI 点「初始化」→ 表格列出现 `temp`，待测面板出现三个参数大字，可视化目标单选钮正常。
5. 手动把 `experiments/3parm_test/config.yaml` 的 `high: 220` 改成 `high: 200` 保存；下拉切到别的再切回 `3parm_test` → 弹「配置冲突」错误且不加载（回退选择）。
6. 切回 `default` → 仍显示 2 参数布局与原数据。

- [ ] **Step 5: 提交**

```bash
git add fdm_mobo_gui.py tests/test_gui_smoke.py
git commit -m "feat: GUI experiment switcher + config-driven rebuild"
```

---

## Self-Review

**Spec coverage:**
- 磁盘布局（experiments/<name>/config.yaml+trials.csv+meta.json，.current）→ Task 2/4。✓
- 冲突检测（指纹 + 拒绝加载 + 文案）→ Task 1（fingerprint）/Task 2（check_conflict）/Task 5（CLI 拒绝）/Task 6（GUI 弹错回退）。✓
- 核心彻底重构（Config/Experiment，去全局）→ Task 1–3。✓
- GUI 下拉 + 新建 + 按配置重建 → Task 6。✓
- 迁移现有数据 → Task 4。✓
- CLI `--exp` → Task 5。✓
- 测试策略 → 每个 Task 含 pytest。✓
- YAGNI（不迁移补值、不内嵌 yaml 编辑器、无对比视图）→ 计划未引入。✓

**Placeholder scan:** 无 TBD/TODO；所有代码步骤含完整代码。✓

**Type consistency:** `Experiment` 方法名（load_trials/save_trials/is_complete/next_idx/check_conflict/init_sobol/to_XY/fit_model/suggest_next/pareto_and_hv/write_meta）在 Task 2/3/5/6 一致；`Config`（from_yaml/from_dict/fieldnames/fingerprint/bounds_tensor）一致；模块函数（list_experiments/get_current/set_current/create_experiment/migrate_legacy/resolve_experiment）签名一致。✓

注：Task 3 的 `to_XY` 调用 `load_trials_done()`，该 helper 在同一 Step 3 代码块中定义，无悬挂引用。
