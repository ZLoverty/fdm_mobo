#!/usr/bin/env python3
"""
fdm_mobo_gui.py — FDM 贝叶斯优化 GUI 前端

依赖: tkinter (标准库), matplotlib
运行: python fdm_mobo_gui.py
"""
from __future__ import annotations

import math
import os
import sys
import threading
import tkinter as tk
import tkinter.simpledialog as simpledialog
from datetime import datetime
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 自动选第一个可用的 CJK 字体（macOS / Linux / Windows 均适用）
_CJK_CANDIDATES = [
    "PingFang SC", "STHeiti", "Heiti SC",   # macOS
    "Microsoft YaHei", "SimHei", "SimSun",   # Windows
    "Noto Sans CJK SC", "WenQuanYi Micro Hei",  # Linux
    "Arial Unicode MS",
]
_available = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示为方块的问题
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── 确定工作目录：打包后用可执行文件所在目录，开发时用脚本目录 ────
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_APP_DIR)
sys.path.insert(0, _APP_DIR)
import fdm_mobo as core


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FDM 多目标贝叶斯优化")
        self.geometry("1300x840")
        self.minsize(960, 640)
        self._pareto_idxs: set[int] = set()
        self.rows: list[dict] = []
        self._cbar: object | None = None   # colorbar 引用，防止重复叠加
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

    # ──────────────────────────── layout ─────────────────────────
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

    # ─────────────────────── 实验切换 / 新建 / 重建 ───────────────
    def _load_experiment(self, name: str) -> None:
        try:
            exp = core.Experiment(core.EXPERIMENTS_DIR / name)
        except Exception as e:
            messagebox.showerror("配置错误", f"无法加载实验 '{name}':\n{e}")
            self._expvar.set(self.exp.dir.name)
            return
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
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
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
        except Exception as e:
            messagebox.showwarning("提示", f"已创建实验,但无法自动打开配置文件:\n{e}\n请用『打开配置』手动编辑。")
        messagebox.showinfo("新建实验",
            "已复制当前配置为模板并打开。\n编辑好维度后回来点『初始化』生成起始点。")

    def _open_config(self) -> None:
        try:
            os.startfile(str(self.exp.dir / "config.yaml"))  # noqa
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    # ────────────────────────── 历史表格 ─────────────────────────
    def _build_table(self, parent: ttk.Frame) -> None:
        hdr = ttk.Frame(parent)
        hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hdr, text="实验历史", font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, text="  双击参数/结果列可编辑",
                  foreground="#888", font=("", 9)).pack(side=tk.LEFT)

        cols = (["idx", "phase"]
                + [p.name for p in self.cfg.params]
                + [o.name for o in self.cfg.objectives]
                + ["time"])
        self._cols = cols

        f = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True, pady=2)

        self.tree = ttk.Treeview(f, columns=cols, show="headings")
        wsz = {"idx": 38, "phase": 56, "time": 126}
        for c in cols:
            self.tree.heading(c, text=c, anchor="center")
            self.tree.column(c, width=wsz.get(c, 72), anchor="center", minwidth=40)
        self.tree.tag_configure("pending", foreground="#b84800")
        self.tree.tag_configure("pareto",  foreground="#0044bb", font=("", 10, "bold"))

        vsb = ttk.Scrollbar(f, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<Double-1>", self._on_dblclick)

    # ─────────────────────── 当前待测面板 ────────────────────────
    def _build_current(self, parent: ttk.Frame) -> None:
        frm = ttk.LabelFrame(parent, text=" 当前待测 ", padding=10)
        frm.pack(fill=tk.X, pady=(0, 6))

        # 参数值大字展示
        pf = ttk.Frame(frm)
        pf.pack(fill=tk.X, pady=(0, 8))
        self._plabels: dict[str, ttk.Label] = {}
        for i, p in enumerate(self.cfg.params):
            col = ttk.Frame(pf)
            col.grid(row=0, column=i, padx=20, sticky="n")
            ttk.Label(col, text=p.name, font=("", 9, "bold")).pack()
            lbl = ttk.Label(col, text="—", font=("", 22, "bold"), foreground="#003399")
            lbl.pack()
            ttk.Label(col, text=f"[{p.low:.4g} – {p.high:.4g}]",
                      foreground="#777", font=("", 8)).pack()
            self._plabels[p.name] = lbl

        # 结果输入
        rf = ttk.LabelFrame(frm, text="填写测量结果", padding=8)
        rf.pack(fill=tk.X, pady=4)
        self._rvars: dict[str, tk.StringVar] = {}
        for i, o in enumerate(self.cfg.objectives):
            tag = "↑ 越大越好" if o.goal == "max" else "↓ 越小越好"
            ttk.Label(rf, text=f"{o.name}  ({tag}):").grid(
                row=i, column=0, sticky="e", padx=6, pady=4)
            var = tk.StringVar()
            entry = ttk.Entry(rf, textvariable=var, width=14, font=("", 11))
            entry.grid(row=i, column=1, padx=6, pady=4, sticky="w")
            entry.bind("<Return>", lambda _e: self._submit())
            self._rvars[o.name] = var

        # 操作按钮
        bf = ttk.Frame(frm)
        bf.pack(fill=tk.X, pady=(10, 0))
        self.btn_submit  = ttk.Button(bf, text="提交结果",    command=self._submit,  width=12)
        self.btn_suggest = ttk.Button(bf, text="获取 BO 建议", command=self._suggest, width=14)
        self.btn_init    = ttk.Button(bf, text="初始化",      command=self._init,    width=9)
        self.btn_submit .pack(side=tk.LEFT, padx=3)
        self.btn_suggest.pack(side=tk.LEFT, padx=3)
        # self.btn_init   .pack(side=tk.LEFT, padx=3)

        self._statusvar = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self._statusvar, foreground="#555",
                  font=("", 9)).pack(anchor="w", pady=(6, 0))
        self._cur_idx: int | None = None

    # ─────────────────────────── 统计面板 ────────────────────────
    def _build_stats(self, parent: ttk.Frame) -> None:
        frm = ttk.LabelFrame(parent, text=" 统计 ", padding=8)
        frm.pack(fill=tk.X)
        self._statsvar = tk.StringVar(value="—")
        ttk.Label(frm, textvariable=self._statsvar, justify=tk.LEFT,
                  font=("Courier", 9)).pack(anchor="w")

    # ─────────────────────────── 可视化面板 ──────────────────────
    def _build_viz(self, parent: ttk.LabelFrame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(anchor="w", pady=(0, 4))
        ttk.Label(ctrl, text="参数空间颜色映射:").pack(side=tk.LEFT, padx=2)
        self._vizvar = tk.StringVar(value=self.cfg.objectives[0].name)
        for o in self.cfg.objectives:
            ttk.Radiobutton(ctrl, text=o.name, variable=self._vizvar,
                            value=o.name, command=self.update_viz).pack(side=tk.LEFT, padx=5)

        self._fig, (self._ax1, self._ax2) = plt.subplots(
            1, 2, figsize=(10, 3.6), constrained_layout=True)
        self._fig.patch.set_facecolor("#f7f7f7")
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────── 数据刷新 ────────────────────────
    def refresh(self) -> None:
        self.rows = self.exp.load_trials()
        self._pareto_idxs = set()
        done = [r for r in self.rows if self.exp.is_complete(r)]
        if len(done) >= 2:
            try:
                p, _ = self.exp.pareto_and_hv()
                self._pareto_idxs = {r["idx"] for r in p}
            except Exception:
                pass
        self._refresh_table()
        self._refresh_current()
        self._refresh_stats()
        self.update_viz()

    def _refresh_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for r in self.rows:
            vals: list[str] = []
            for c in self._cols:
                v = r.get(c, "")
                if c in {o.name for o in self.cfg.objectives}:
                    vals.append("?" if (isinstance(v, float) and math.isnan(v))
                                else f"{v:g}" if isinstance(v, float) else str(v))
                else:
                    vals.append(f"{v:g}" if isinstance(v, float) else str(v) if v is not None else "")
            tag = ("pareto"  if r["idx"] in self._pareto_idxs else
                   "pending" if not self.exp.is_complete(r) else "")
            self.tree.insert("", "end", iid=str(r["idx"]), values=vals, tags=(tag,))

    def _refresh_current(self) -> None:
        pending = [r for r in self.rows if not self.exp.is_complete(r)]
        done    = [r for r in self.rows if self.exp.is_complete(r)]
        if pending:
            r = pending[0]
            self._cur_idx = r["idx"]
            for p in self.cfg.params:
                self._plabels[p.name].config(text=f"{r[p.name]:g}")
            for o in self.cfg.objectives:
                self._rvars[o.name].set("")
            self.btn_submit .config(state="normal")
            self.btn_suggest.config(state="disabled")
            self._statusvar.set(f"#{r['idx']} 待测  |  共 {len(pending)} 个未完成")
        else:
            self._cur_idx = None
            for p in self.cfg.params:
                self._plabels[p.name].config(text="—")
            self.btn_submit.config(state="disabled")
            if not self.rows:
                self.btn_suggest.config(state="disabled")
                self._statusvar.set("尚无数据，点击「初始化」开始")
            elif len(done) < 2:
                self.btn_suggest.config(state="disabled")
                self._statusvar.set(f"已完成 {len(done)} 个，至少需要 2 个才能运行 BO")
            else:
                self.btn_suggest.config(state="normal")
                self._statusvar.set(f"全部已完成（{len(done)} 个），可获取 BO 建议")

    def _refresh_stats(self) -> None:
        done = [r for r in self.rows if self.exp.is_complete(r)]
        if not done:
            self._statsvar.set("暂无已完成实验")
            return
        lines = [f"总计 {len(self.rows)} 点  |  完成 {len(done)}  |  待测 {len(self.rows)-len(done)}"]
        if len(done) >= 2:
            try:
                _, hv = self.exp.pareto_and_hv()
                lines.append(f"超体积 = {hv:.4g}  |  Pareto = {len(self._pareto_idxs)} 个非支配点")
            except Exception:
                pass
        for o in self.cfg.objectives:
            vs = [r[o.name] for r in done]
            best = max(vs) if o.goal == "max" else min(vs)
            lines.append(f"最佳 {o.name} = {best:g}")
        self._statsvar.set("\n".join(lines))

    # ─────────────────────────── 可视化绘图 ──────────────────────
    def update_viz(self) -> None:
        self._ax1.clear()
        self._ax2.clear()

        # 移除旧 colorbar，避免重叠缩小
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        rows    = self.rows
        oname   = self._vizvar.get()
        done    = [r for r in rows if self.exp.is_complete(r)]
        pending = [r for r in rows if not self.exp.is_complete(r)]

        # ── 左图：参数空间 ─────────────────────────────────────
        ax = self._ax1
        if len(self.cfg.params) >= 2:
            px, py  = self.cfg.params[0], self.cfg.params[1]
            if done:
                xs = np.array([r[px.name] for r in done])
                ys = np.array([r[py.name] for r in done])
                vs = np.array([r[oname]   for r in done], dtype=float)
                sc = ax.scatter(xs, ys, c=vs, cmap="RdYlGn", s=75, zorder=3,
                                edgecolors="#444", linewidths=0.5)
                self._cbar = self._fig.colorbar(sc, ax=ax, label=oname, shrink=0.85, pad=0.02)
                for r in done:
                    ax.annotate(f"#{r['idx']}", (r[px.name], r[py.name]),
                                xytext=(4, 3), textcoords="offset points",
                                fontsize=7.5, color="#222")
                # Pareto 点加蓝圈标记
                pp = [r for r in done if r["idx"] in self._pareto_idxs]
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

        # ── 右图：目标空间 ─────────────────────────────────────
        ax2 = self._ax2
        if len(self.cfg.objectives) >= 2 and done:
            o1, o2 = self.cfg.objectives[0], self.cfg.objectives[1]
            x2 = np.array([r[o1.name] for r in done])
            y2 = np.array([r[o2.name] for r in done])
            ax2.scatter(x2, y2, s=68, color="#2e8b2e", zorder=3,
                        edgecolors="#333", linewidths=0.5, label="已测")
            for r in done:
                ax2.annotate(f"#{r['idx']}", (r[o1.name], r[o2.name]),
                             xytext=(4, 3), textcoords="offset points", fontsize=7.5)
            pp = [r for r in done if r["idx"] in self._pareto_idxs]
            if pp:
                ax2.scatter([r[o1.name] for r in pp], [r[o2.name] for r in pp],
                            s=160, facecolors="none", edgecolors="#0044bb",
                            linewidths=2.2, zorder=5, label="Pareto 前沿")
                # Pareto 阶梯连线（当两者均最大化时成立）
                if len(pp) > 1:
                    sp = sorted(pp, key=lambda r: r[o1.name])
                    ax2.step([r[o1.name] for r in sp], [r[o2.name] for r in sp],
                             where="post", color="#0044bb", alpha=0.4, lw=1.6, zorder=4)
        ax2.set_xlabel(self.cfg.objectives[0].name, fontsize=10)
        ax2.set_ylabel(self.cfg.objectives[1].name if len(self.cfg.objectives) >= 2 else "", fontsize=10)
        ax2.set_title("目标空间 (Pareto 前沿)", fontsize=10)
        ax2.grid(True, alpha=0.22)
        if done:
            ax2.legend(fontsize=8, loc="lower right")

        self._canvas.draw()

    # ────────────────────────── 操作 ─────────────────────────────
    def _submit(self) -> None:
        if self._cur_idx is None:
            return
        vals: dict[str, float] = {}
        for o in self.cfg.objectives:
            s = self._rvars[o.name].get().strip()
            if not s:
                messagebox.showwarning("缺少数据", f"请填写 {o.name} 的测量值")
                return
            try:
                vals[o.name] = float(s)
            except ValueError:
                messagebox.showerror("格式错误", f"{o.name} 必须是数字")
                return
        rows = self.exp.load_trials()
        for r in rows:
            if r["idx"] == self._cur_idx:
                for o in self.cfg.objectives:
                    r[o.name] = vals[o.name]
                r["time"] = datetime.now().isoformat(timespec="seconds")
                break
        self.exp.save_trials(rows)
        self.refresh()

    def _suggest(self) -> None:
        self.btn_suggest.config(state="disabled", text="计算中…")
        self._statusvar.set("正在运行贝叶斯优化，请稍候…")

        def _run() -> None:
            try:
                rows = self.exp.load_trials()
                cand = self.exp.suggest_next()
                start = self.exp.next_idx(rows)
                for k, c in enumerate(cand):
                    r: dict = {"idx": start + k, "phase": "bo", "time": ""}
                    for j, par in enumerate(self.cfg.params):
                        r[par.name] = round(float(c[j]), 3)
                    for o in self.cfg.objectives:
                        r[o.name] = math.nan
                    rows.append(r)
                self.exp.save_trials(rows)
                self.after(0, self.refresh)
                self.after(0, lambda: self._statusvar.set("BO 建议已生成"))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: messagebox.showerror("BO 错误", msg))
                self.after(0, self.refresh)
            finally:
                self.after(0, lambda: self.btn_suggest.config(text="获取 BO 建议"))

        threading.Thread(target=_run, daemon=True).start()

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

    # ────────────── 双击表格编辑 ─────────────────────────────────
    def _on_dblclick(self, event: tk.Event) -> None:
        if self.tree.identify("region", event.x, event.y) != "cell":
            return
        col_idx  = int(self.tree.identify_column(event.x).lstrip("#")) - 1
        row_id   = self.tree.identify_row(event.y)
        if not row_id:
            return
        col_name = self._cols[col_idx]

        editable = {o.name for o in self.cfg.objectives} | {p.name for p in self.cfg.params}
        if col_name not in editable:
            return

        vals    = self.tree.item(row_id)["values"]
        idx     = int(vals[0])
        cur_val = vals[col_idx]

        new_s = simpledialog.askstring(
            "编辑数值",
            f"#{idx}  {col_name}\n当前值: {cur_val}",
            parent=self,
        )
        if not new_s or not new_s.strip():
            return
        try:
            fv = float(new_s.strip())
        except ValueError:
            messagebox.showerror("格式错误", "请输入数字")
            return

        rows = self.exp.load_trials()
        for r in rows:
            if r["idx"] == idx:
                r[col_name] = fv
                if self.exp.is_complete(r):
                    r["time"] = datetime.now().isoformat(timespec="seconds")
                break
        self.exp.save_trials(rows)
        self.refresh()


if __name__ == "__main__":
    App().mainloop()
