#!/usr/bin/env python3
"""
fdm_mobo_web.py — FDM 多目标贝叶斯优化 Web 前端

把原 tkinter GUI(fdm_mobo_gui.py) 封装为 Web App，界面布局保持基本不变：
顶部实验选择栏 + 左侧历史表格 + 右侧当前待测/统计 + 底部可视化双图。
新增：数据下载接口（trials.csv / config.yaml）。

依赖: pip install flask matplotlib pyyaml  (BO 操作另需 botorch，懒加载)
运行: python fdm_mobo_web.py    然后浏览器打开 http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务端渲染 PNG
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 自动选第一个可用的 CJK 字体
_CJK_CANDIDATES = [
    "PingFang SC", "STHeiti", "Heiti SC",
    "Microsoft YaHei", "SimHei", "SimSun",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
_available = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from flask import (
    Flask, jsonify, request, send_file, render_template, abort, Response,
)

# ── 工作目录：脚本目录，便于相对路径定位 experiments/ ───────────────
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_APP_DIR)
sys.path.insert(0, _APP_DIR)

import fdm_mobo as core

app = Flask(__name__)


# ─────────────────────────── 辅助 ────────────────────────────────────
def _ensure_some_experiment() -> None:
    if not core.list_experiments():
        core.create_experiment("default", core.LEGACY_CONFIG_YAML)
        core.set_current("default")


def _current_name() -> str:
    return core.get_current() or (core.list_experiments() or ["default"])[0]


def _open_experiment(name: str | None) -> core.Experiment:
    """打开指定实验（默认当前），并做配置冲突检测。"""
    name = name or _current_name()
    if name not in core.list_experiments():
        abort(404, description=f"实验 '{name}' 不存在")
    exp = core.Experiment(core.EXPERIMENTS_DIR / name)
    conflict = exp.check_conflict()
    if conflict:
        abort(409, description=conflict)
    return exp


def _pareto_idxs(exp: core.Experiment, rows: list[dict]) -> set[int]:
    done = [r for r in rows if exp.is_complete(r)]
    if len(done) < 2:
        return set()
    try:
        p, _ = exp.pareto_and_hv()
        return {r["idx"] for r in p}
    except Exception:
        return set()


def _row_to_json(exp: core.Experiment, r: dict) -> dict:
    """把一行 trial 转成 JSON 友好的字典（NaN -> null）。"""
    out: dict = {"idx": r["idx"], "phase": r.get("phase", ""), "time": r.get("time", "")}
    for p in exp.cfg.params:
        out[p.name] = r[p.name]
    for o in exp.cfg.objectives:
        v = r[o.name]
        out[o.name] = None if (isinstance(v, float) and math.isnan(v)) else v
    out["_complete"] = exp.is_complete(r)
    return out


# ─────────────────────────── 页面 ────────────────────────────────────
@app.route("/")
def index():
    _ensure_some_experiment()
    return render_template("index.html")


# ─────────────────────────── API：实验管理 ───────────────────────────
@app.route("/api/experiments")
def api_experiments():
    _ensure_some_experiment()
    return jsonify({
        "experiments": core.list_experiments(),
        "current": _current_name(),
    })


@app.route("/api/experiments/select", methods=["POST"])
def api_select():
    name = (request.json or {}).get("name", "")
    if name not in core.list_experiments():
        abort(404, description=f"实验 '{name}' 不存在")
    exp = core.Experiment(core.EXPERIMENTS_DIR / name)
    conflict = exp.check_conflict()
    if conflict:
        abort(409, description=conflict)
    core.set_current(name)
    return jsonify({"ok": True, "current": name})


@app.route("/api/experiments/new", methods=["POST"])
def api_new():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        abort(400, description="实验名不能为空")
    if name in core.list_experiments():
        abort(409, description=f"实验 '{name}' 已存在")
    # 以当前实验的 config 作为模板
    cur = _open_experiment(None)
    template = (cur.dir / "config.yaml").read_text(encoding="utf-8")
    try:
        core.create_experiment(name, template)
    except ValueError as e:
        abort(400, description=str(e))
    core.set_current(name)
    return jsonify({"ok": True, "current": name})


# ─────────────────────────── API：配置 ───────────────────────────────
# 这些是 BO 算法/随机数内部参数，不在表单中展示，保存时原样保留。
_HIDDEN_CFG_KEYS = ("seed", "num_restarts", "raw_samples", "mc_samples")
# YAML 文件中键的输出顺序
_CFG_KEY_ORDER = ("params", "objectives", "n_init", "seed", "batch",
                  "num_restarts", "raw_samples", "mc_samples")


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    exp = _open_experiment(request.args.get("exp"))
    if request.method == "GET":
        return jsonify({
            "name": exp.dir.name,
            "params": [{"name": p.name, "low": p.low, "high": p.high} for p in exp.cfg.params],
            "objectives": [{"name": o.name, "goal": o.goal} for o in exp.cfg.objectives],
            "n_init": exp.cfg.n_init,
            "batch": exp.cfg.batch,
        })

    # POST：从结构化表单保存配置，隐藏的算法参数沿用旧值
    import yaml
    data = request.json or {}
    old = yaml.safe_load((exp.dir / "config.yaml").read_text(encoding="utf-8")) or {}

    params, objectives = [], []
    for p in data.get("params", []):
        name = (p.get("name") or "").strip()
        if not name:
            abort(400, description="参数名不能为空")
        try:
            low, high = float(p["low"]), float(p["high"])
        except (TypeError, ValueError, KeyError):
            abort(400, description=f"参数 '{name}' 的范围必须是数字")
        if high <= low:
            abort(400, description=f"参数 '{name}' 的上限必须大于下限")
        params.append({"name": name, "low": low, "high": high})
    for o in data.get("objectives", []):
        name = (o.get("name") or "").strip()
        goal = o.get("goal")
        if not name:
            abort(400, description="目标名不能为空")
        if goal not in ("max", "min"):
            abort(400, description=f"目标 '{name}' 的优化方向必须是 max 或 min")
        objectives.append({"name": name, "goal": goal})
    if not params:
        abort(400, description="至少需要 1 个参数")
    if not objectives:
        abort(400, description="至少需要 1 个目标")

    new_cfg = dict(old)
    new_cfg["params"] = params
    new_cfg["objectives"] = objectives
    for k in ("n_init", "batch"):
        if k in data and data[k] not in ("", None):
            try:
                new_cfg[k] = int(data[k])
            except (TypeError, ValueError):
                abort(400, description=f"{k} 必须是整数")

    try:
        core.Config.from_dict(new_cfg)  # 校验
    except Exception as e:
        abort(400, description=f"配置无效: {e}")

    ordered = {k: new_cfg[k] for k in _CFG_KEY_ORDER if k in new_cfg}
    for k, v in new_cfg.items():  # 保留任何未知键
        ordered.setdefault(k, v)
    text = yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False,
                          default_flow_style=False)
    (exp.dir / "config.yaml").write_text(text, encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/config/raw", methods=["GET", "POST"])
def api_config_raw():
    exp = _open_experiment(request.args.get("exp"))
    path = exp.dir / "config.yaml"
    if request.method == "GET":
        return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")
    # POST：保存新的 yaml（先校验可解析）
    text = (request.json or {}).get("text", "")
    try:
        import yaml
        core.Config.from_dict(yaml.safe_load(text))
    except Exception as e:
        abort(400, description=f"配置无法解析: {e}")
    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True})


# ─────────────────────────── API：数据 ───────────────────────────────
@app.route("/api/trials")
def api_trials():
    exp = _open_experiment(request.args.get("exp"))
    rows = exp.load_trials()
    pareto = _pareto_idxs(exp, rows)
    done = [r for r in rows if exp.is_complete(r)]

    # 统计
    stats: dict = {
        "total": len(rows),
        "done": len(done),
        "pending": len(rows) - len(done),
        "hypervolume": None,
        "pareto_count": len(pareto),
        "best": {},
    }
    if len(done) >= 2:
        try:
            _, hv = exp.pareto_and_hv()
            stats["hypervolume"] = hv
        except Exception:
            pass
    for o in exp.cfg.objectives:
        vs = [r[o.name] for r in done]
        if vs:
            stats["best"][o.name] = max(vs) if o.goal == "max" else min(vs)

    # 当前待测点
    pending = [r for r in rows if not exp.is_complete(r)]
    current = _row_to_json(exp, pending[0]) if pending else None

    return jsonify({
        "name": exp.dir.name,
        "params": [{"name": p.name, "low": p.low, "high": p.high} for p in exp.cfg.params],
        "objectives": [{"name": o.name, "goal": o.goal} for o in exp.cfg.objectives],
        "rows": [_row_to_json(exp, r) for r in rows],
        "pareto": sorted(pareto),
        "current": current,
        "pending_count": len(pending),
        "stats": stats,
    })


@app.route("/api/submit", methods=["POST"])
def api_submit():
    exp = _open_experiment(request.args.get("exp"))
    data = request.json or {}
    idx = data.get("idx")
    values = data.get("values", {})
    rows = exp.load_trials()
    target = next((r for r in rows if r["idx"] == idx), None)
    if target is None:
        abort(404, description=f"找不到 #{idx}")
    for o in exp.cfg.objectives:
        if o.name not in values or values[o.name] in ("", None):
            abort(400, description=f"请填写 {o.name} 的测量值")
        try:
            target[o.name] = float(values[o.name])
        except (TypeError, ValueError):
            abort(400, description=f"{o.name} 必须是数字")
    target["time"] = datetime.now().isoformat(timespec="seconds")
    exp.save_trials(rows)
    return jsonify({"ok": True})


@app.route("/api/edit", methods=["POST"])
def api_edit():
    """双击编辑：修改某行的某个参数/目标值。"""
    exp = _open_experiment(request.args.get("exp"))
    data = request.json or {}
    idx = data.get("idx")
    col = data.get("col")
    editable = {o.name for o in exp.cfg.objectives} | {p.name for p in exp.cfg.params}
    if col not in editable:
        abort(400, description=f"列 '{col}' 不可编辑")
    try:
        fv = float(data.get("value"))
    except (TypeError, ValueError):
        abort(400, description="请输入数字")
    rows = exp.load_trials()
    target = next((r for r in rows if r["idx"] == idx), None)
    if target is None:
        abort(404, description=f"找不到 #{idx}")
    target[col] = fv
    if exp.is_complete(target):
        target["time"] = datetime.now().isoformat(timespec="seconds")
    exp.save_trials(rows)
    return jsonify({"ok": True})


@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    exp = _open_experiment(request.args.get("exp"))
    rows = exp.load_trials()
    if len([r for r in rows if exp.is_complete(r)]) < 2:
        abort(400, description="已完成的实验少于 2 个，至少需要 2 个才能运行 BO")
    if [r for r in rows if not exp.is_complete(r)]:
        abort(400, description="还有未测量的点，先提交结果再获取建议")
    try:
        cand = exp.suggest_next()
    except Exception as e:
        abort(500, description=f"BO 错误: {e}")
    start = exp.next_idx(rows)
    for k, c in enumerate(cand):
        r: dict = {"idx": start + k, "phase": "bo", "time": ""}
        for j, par in enumerate(exp.cfg.params):
            r[par.name] = round(float(c[j]), 3)
        for o in exp.cfg.objectives:
            r[o.name] = math.nan
        rows.append(r)
    exp.save_trials(rows)
    return jsonify({"ok": True, "added": len(cand)})


@app.route("/api/init", methods=["POST"])
def api_init():
    exp = _open_experiment(request.args.get("exp"))
    exp = core.Experiment(exp.dir)  # 重新读最新 config.yaml
    if exp.trials_path.exists():
        exp.trials_path.unlink()
    try:
        exp.init_sobol()
    except Exception as e:
        abort(500, description=f"初始化失败: {e}")
    return jsonify({"ok": True})


# ─────────────────────────── 可视化（服务端渲染） ────────────────────
@app.route("/api/plot.png")
def api_plot():
    exp = _open_experiment(request.args.get("exp"))
    oname = request.args.get("color") or exp.cfg.objectives[0].name
    rows = exp.load_trials()
    pareto_idxs = _pareto_idxs(exp, rows)
    done = [r for r in rows if exp.is_complete(r)]
    pending = [r for r in rows if not exp.is_complete(r)]
    if oname not in {o.name for o in exp.cfg.objectives}:
        oname = exp.cfg.objectives[0].name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f7")

    # ── 左图：参数空间 ──
    ax = ax1
    if len(exp.cfg.params) >= 2:
        px, py = exp.cfg.params[0], exp.cfg.params[1]
        if done:
            xs = np.array([r[px.name] for r in done])
            ys = np.array([r[py.name] for r in done])
            vs = np.array([r[oname] for r in done], dtype=float)
            sc = ax.scatter(xs, ys, c=vs, cmap="RdYlGn", s=75, zorder=3,
                            edgecolors="#444", linewidths=0.5)
            fig.colorbar(sc, ax=ax, label=oname, shrink=0.85, pad=0.02)
            for r in done:
                ax.annotate(f"#{r['idx']}", (r[px.name], r[py.name]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color="#222")
            pp = [r for r in done if r["idx"] in pareto_idxs]
            if pp:
                ax.scatter([r[px.name] for r in pp], [r[py.name] for r in pp],
                           s=200, facecolors="none", edgecolors="#0044bb",
                           linewidths=2.2, zorder=5, label="Pareto")
        if pending:
            ax.scatter([r[px.name] for r in pending], [r[py.name] for r in pending],
                       marker="x", s=95, c="#b84800", lw=2.2, zorder=4, label="待测")
            for r in pending:
                ax.annotate(f"#{r['idx']}", (r[px.name], r[py.name]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color="#b84800")
        mx = (px.high - px.low) * 0.06
        my = (py.high - py.low) * 0.06
        ax.set_xlim(px.low - mx, px.high + mx)
        ax.set_ylim(py.low - my, py.high + my)
        ax.set_xlabel(px.name, fontsize=10)
        ax.set_ylabel(py.name, fontsize=10)
        ax.set_title("参数空间", fontsize=10)
        ax.grid(True, alpha=0.22)
        if done or pending:
            ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "需要 ≥2 个参数才能绘制参数空间",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888")
        ax.set_title("参数空间", fontsize=10)

    # ── 右图：目标空间 ──
    if len(exp.cfg.objectives) >= 2 and done:
        o1, o2 = exp.cfg.objectives[0], exp.cfg.objectives[1]
        x2 = np.array([r[o1.name] for r in done])
        y2 = np.array([r[o2.name] for r in done])
        ax2.scatter(x2, y2, s=68, color="#2e8b2e", zorder=3,
                    edgecolors="#333", linewidths=0.5, label="已测")
        for r in done:
            ax2.annotate(f"#{r['idx']}", (r[o1.name], r[o2.name]),
                         xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        pp = [r for r in done if r["idx"] in pareto_idxs]
        if pp:
            ax2.scatter([r[o1.name] for r in pp], [r[o2.name] for r in pp],
                        s=160, facecolors="none", edgecolors="#0044bb",
                        linewidths=2.2, zorder=5, label="Pareto 前沿")
            if len(pp) > 1:
                sp = sorted(pp, key=lambda r: r[o1.name])
                ax2.step([r[o1.name] for r in sp], [r[o2.name] for r in sp],
                         where="post", color="#0044bb", alpha=0.4, lw=1.6, zorder=4)
    ax2.set_xlabel(exp.cfg.objectives[0].name, fontsize=10)
    ax2.set_ylabel(exp.cfg.objectives[1].name if len(exp.cfg.objectives) >= 2 else "", fontsize=10)
    ax2.set_title("目标空间 (Pareto 前沿)", fontsize=10)
    ax2.grid(True, alpha=0.22)
    if done:
        ax2.legend(fontsize=8, loc="lower right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ─────────────────────────── 数据下载接口 ────────────────────────────
@app.route("/api/download/trials")
def api_download_trials():
    exp = _open_experiment(request.args.get("exp"))
    if not exp.trials_path.exists():
        abort(404, description="该实验尚无 trials.csv")
    return send_file(
        exp.trials_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{exp.dir.name}_trials.csv",
    )


@app.route("/api/download/config")
def api_download_config():
    exp = _open_experiment(request.args.get("exp"))
    return send_file(
        exp.dir / "config.yaml",
        mimetype="text/yaml",
        as_attachment=True,
        download_name=f"{exp.dir.name}_config.yaml",
    )


# ─────────────────────────── 错误处理 ────────────────────────────────
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(500)
def handle_error(e):
    return jsonify({"error": getattr(e, "description", str(e))}), e.code


if __name__ == "__main__":
    _ensure_some_experiment()
    port = int(os.environ.get("PORT", "5000"))
    print(f"FDM MOBO Web 启动中… 打开 http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
