#!/usr/bin/env python3
"""
fdm_mobo.py — FDM 双目标贝叶斯优化框架(human-in-the-loop)

    输入 x = (fan 风扇功率%, flow 流量比例%)
    目标 y = (surface 表面质量, TS 拉伸强度)

工作流(每个"实验"= 打印一个件 + 测表面 + 拉伸测试,通常跨天):
    1. init       生成初始 Sobol 实验点,写入 trials.csv(测量值留空)
    2. (离线)    照建议的参数打印 + 测量
    3. record     回填测得的 surface 与 TS
    4. suggest    拟合 GP + qLogNEHVI,给出下一个实验点(再回到 2)
    status        查看全部实验 / 当前 Pareto 前沿 / 超体积(hypervolume)
    apply         (可选)把某个点的 fan/flow 通过 Moonraker 下发到打印机

设计要点:
    * 所有状态只存在 trials.csv —— 可随时关掉程序、几天后回来继续。
    * 内部把两个目标统一成"都最大化"(min 目标乘 -1),GP / 采集函数只管最大化。
    * 输入按搜索范围归一化、输出做标准化(GP 的前提),范围改 PARAMS 即可。

依赖: pip install botorch        # 会带上 torch / gpytorch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

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


# ============================== 张量 / 模型 ==============================

def bounds_tensor() -> torch.Tensor:
    return torch.tensor([[p.low for p in PARAMS], [p.high for p in PARAMS]])


def to_XY(rows: list[dict]):
    """只取已完成的实验,返回 (X 原始单位, Y 内部最大化空间, 已完成行)。"""
    done = [r for r in rows if is_complete(r)]
    X = torch.tensor([[r[p.name] for p in PARAMS] for r in done])
    Y_raw = torch.tensor([[r[o.name] for o in OBJECTIVES] for r in done])
    signs = torch.tensor([o.sign for o in OBJECTIVES])
    Y = Y_raw * signs  # min 目标取负,内部一律最大化
    return X, Y, done


def fit_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """一个多输出 GP 同时建模两个目标。输入按搜索范围归一化,输出标准化。"""
    model = SingleTaskGP(
        X, Y,
        input_transform=Normalize(d=X.shape[-1], bounds=bounds_tensor()),
        outcome_transform=Standardize(m=Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def ref_point(Y: torch.Tensor) -> torch.Tensor:
    """参考点取比最差观测再差一点(内部最大化空间:每维最小值再减 10% 量程)。"""
    mn = Y.min(dim=0).values
    rng = Y.max(dim=0).values - mn
    rng = torch.where(rng > 0, rng, torch.ones_like(rng))
    return mn - 0.1 * rng


def suggest_next(rows: list[dict]) -> torch.Tensor:
    X, Y, done = to_XY(rows)
    if len(done) < 2:
        raise SystemExit("已完成的实验少于 2 个,先 record 更多点再 suggest。")
    model = fit_model(X, Y)
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point(Y).tolist(),
        X_baseline=X,                       # 原始单位,模型内部会归一化
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
    )
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor(),
        q=BATCH,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    return candidates.detach()


def pareto_and_hv(rows: list[dict]):
    X, Y, done = to_XY(rows)
    if len(done) == 0:
        return [], float("nan")
    mask = is_non_dominated(Y)
    hv = DominatedPartitioning(ref_point=ref_point(Y), Y=Y).compute_hypervolume().item()
    pareto = [done[i] for i in range(len(done)) if bool(mask[i])]
    return pareto, hv


# ============================== 展示 ==============================

def fmt_params(r: dict) -> str:
    return "  ".join(f"{p.name}={r[p.name]:g}" for p in PARAMS)


def fmt_objs(r: dict) -> str:
    return "  ".join(
        f"{o.name}=" + ("?" if math.isnan(r[o.name]) else f"{r[o.name]:g}")
        for o in OBJECTIVES
    )


def print_rows(rows: list[dict]) -> None:
    for r in rows:
        tag = "[待测]" if not is_complete(r) else "      "
        print(f"  #{r['idx']:<3} {tag} {fmt_params(r):<28} {fmt_objs(r)}")


# ============================== 子命令 ==============================

def cmd_init(args):
    rows = load_trials()
    if rows:
        print("trials.csv 已存在,跳过 init(用 suggest / record 继续)。")
        return
    pts = draw_sobol_samples(bounds=bounds_tensor(), n=N_INIT, q=1, seed=SEED).squeeze(1)
    rows = []
    for i, p in enumerate(pts):
        r = {"idx": i, "phase": "init", "time": ""}
        for j, par in enumerate(PARAMS):
            r[par.name] = round(float(p[j]), 3)
        for o in OBJECTIVES:
            r[o.name] = math.nan
        rows.append(r)
    save_trials(rows)
    print(f"已生成 {N_INIT} 个初始点 -> {TRIALS_CSV}\n按下面参数逐个打印并测量,然后 record:")
    print_rows(rows)


def cmd_suggest(args):
    rows = load_trials()
    if not rows:
        raise SystemExit("还没有 trials.csv,先运行 init。")
    pending = [r for r in rows if not is_complete(r)]
    if pending:
        print("还有未测量的点,先把它们打印+测量并 record,再来 suggest:")
        print_rows(pending)
        return
    cand = suggest_next(rows)
    start = next_idx(rows)
    new = []
    for k, c in enumerate(cand):
        r = {"idx": start + k, "phase": "bo", "time": ""}
        for j, par in enumerate(PARAMS):
            r[par.name] = round(float(c[j]), 3)
        for o in OBJECTIVES:
            r[o.name] = math.nan
        rows.append(r)
        new.append(r)
    save_trials(rows)
    print("下一个实验点(qLogNEHVI):")
    print_rows(new)


def cmd_record(args):
    rows = load_trials()
    by_idx = {r["idx"]: r for r in rows}

    if args.idx is not None:
        r = by_idx.get(args.idx)
        if r is None:
            raise SystemExit(f"找不到 #{args.idx}")
        if args.values:
            if len(args.values) != len(OBJECTIVES):
                raise SystemExit(f"需要 {len(OBJECTIVES)} 个值: {[o.name for o in OBJECTIVES]}")
            for o, v in zip(OBJECTIVES, args.values):
                r[o.name] = float(v)
        else:
            for o in OBJECTIVES:
                v = input(f"#{args.idx} {o.name} = ").strip()
                if v:
                    r[o.name] = float(v)
        r["time"] = datetime.now().isoformat(timespec="seconds")
        save_trials(rows)
        print("已保存 #%d: %s" % (args.idx, fmt_objs(r)))
        return

    # 交互模式:逐个填未测量的点
    pending = [r for r in rows if not is_complete(r)]
    if not pending:
        print("没有待测量的点。")
        return
    for r in pending:
        print(f"\n#{r['idx']}  {fmt_params(r)}  (留空跳过)")
        for o in OBJECTIVES:
            v = input(f"  {o.name} = ").strip()
            if v:
                r[o.name] = float(v)
        if is_complete(r):
            r["time"] = datetime.now().isoformat(timespec="seconds")
    save_trials(rows)
    print("\n已保存。")


def cmd_status(args):
    rows = load_trials()
    if not rows:
        print("还没有数据。先 init。")
        return
    done = [r for r in rows if is_complete(r)]
    print(f"实验总数 {len(rows)},已完成 {len(done)},待测 {len(rows) - len(done)}")
    print_rows(rows)
    if done:
        pareto, hv = pareto_and_hv(rows)
        print(f"\n当前 Pareto 前沿({len(pareto)} 个非支配点):")
        print_rows(pareto)
        print(f"\n超体积 hypervolume = {hv:.4g}（越大越好,用来追踪每轮进展）")


def cmd_apply(args):
    """可选:把某个点的 fan/flow 通过 Moonraker 下发到打印机。
    注意:这只设置风扇与流量,真正起跑打印仍走你 HEPiC / 切片那一套。"""
    rows = load_trials()
    r = {x["idx"]: x for x in rows}.get(args.idx)
    if r is None:
        raise SystemExit(f"找不到 #{args.idx}")
    fan = r["fan"]
    flow = r["flow"]
    fan_s = int(round(fan / 100.0 * 255))
    gcode = f"M106 S{fan_s}\nM221 S{int(round(flow))}"
    print(f"#{args.idx} -> fan {fan:g}% (M106 S{fan_s}), flow {flow:g}% (M221 S{int(round(flow))})")
    if not args.host:
        print("(未提供 --host,仅打印 gcode,不下发)")
        print(gcode)
        return
    import json
    import urllib.request
    url = args.host.rstrip("/") + "/printer/gcode/script"
    req = urllib.request.Request(
        url,
        data=json.dumps({"script": gcode}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("Moonraker:", resp.status, resp.read().decode()[:200])


# ============================== 入口 ==============================

def main():
    ap = argparse.ArgumentParser(description="FDM 双目标贝叶斯优化 (fan, flow) -> (surface, TS)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="生成初始 Sobol 实验点").set_defaults(func=cmd_init)
    sub.add_parser("suggest", help="给出下一个实验点").set_defaults(func=cmd_suggest)
    sub.add_parser("status", help="查看实验 / Pareto / 超体积").set_defaults(func=cmd_status)

    pr = sub.add_parser("record", help="回填测量值")
    pr.add_argument("--idx", type=int, help="指定 trial 编号;不给则交互回填所有待测点")
    pr.add_argument("values", nargs="*", help=f"按顺序给 {[o.name for o in OBJECTIVES]}")
    pr.set_defaults(func=cmd_record)

    pa = sub.add_parser("apply", help="(可选) 经 Moonraker 下发 fan/flow")
    pa.add_argument("--idx", type=int, required=True)
    pa.add_argument("--host", type=str, default="", help="如 http://192.168.1.50")
    pa.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
