# FDM MOBO — 配置外提与实验文件夹设计

日期: 2026-06-25
状态: 已批准（待写实现计划）

## 背景与问题

`fdm_mobo` 是一个 human-in-the-loop 的双目标贝叶斯优化小程序：输入参数（fan, flow），目标（surface, TS），用 BoTorch 跑 qLogNEHVI，状态全存在 `trials.csv`。

当前控制参数与优化目标硬编码在 `fdm_mobo.py` 的 `PARAMS` / `OBJECTIVES` 模块全局变量里（约 74–92 行）。GUI（`fdm_mobo_gui.py`）在构造时直接引用 `core.PARAMS` / `core.OBJECTIVES` 来生成表格列、当前待测面板、可视化控件，维度在构造时被定死。

**目标**：把控制参数和优化目标提取成独立配置文件，方便后续增加优化维度。

**核心矛盾**：`trials.csv` 里的数据只有在生成它的那套维度定义下才有意义。配置与数据天然耦合——一旦改维度，旧数据列对不上、Sobol 起始点也是按旧维度抽的。把配置单独抽出来后，配置文件与数据文件就可能"互相矛盾"，需要明确的冲突解决策略。

## 已确认的关键决策

1. **改维度 = 开新实验，旧数据整套作废**。不做数据迁移/补维度。旧数据归档保留但不参与新模型。
2. **存储布局：每个实验一个文件夹**。配置快照与数据物理同处一处，从根本消除"两文件谁对谁错"。
3. **配置格式：YAML**（加维度 = 加一行，对研究迭代最友好；引入 pyyaml 轻依赖）。
4. **GUI 管理实验：内置下拉框 + 新建按钮**，全程不离开程序。
5. **核心库做彻底重构**：配置从模块全局变量改为可传递的 `Config` / `Experiment` 对象。

## 磁盘布局

```
fdm_mobo/
  fdm_mobo.py            # 核心库（BO + 数据层，去掉硬编码配置）
  fdm_mobo_gui.py        # GUI
  experiments/
    .current             # 一行文本：当前实验文件夹名
    2parm_fan_flow/
      config.yaml        # 维度定义（init 后冻结）
      trials.csv         # 数据
      meta.json          # {config_fingerprint, created}
    3parm_add_temp/
      config.yaml
      trials.csv
      meta.json
```

配置与数据在同一文件夹，在 `init`（生成 Sobol 起始点）那一刻一起诞生，不存在跨文件不一致。

### config.yaml 示例

```yaml
params:
  - {name: fan,  low: 0,   high: 100}
  - {name: flow, low: 0.9, high: 1.1}
  - {name: temp, low: 190, high: 220}   # 加一行就加一维
objectives:
  - {name: surface, goal: max}
  - {name: TS,      goal: max}
n_init: 6
seed: 0
batch: 1
num_restarts: 12
raw_samples: 256
mc_samples: 128
```

## 冲突检测

文件夹内的 `config.yaml` 在 `init` 之后视为**冻结**。

- `init` 时对配置做归一化指纹（hash of：参数名+low+high、目标名+goal、n_init、seed），写入 `meta.json`。
- 每次加载实验，用当前 `config.yaml` 重算指纹与 `meta.json` 比对：
  - **一致** → 正常加载。
  - **不一致**（事后手改了已有数据实验的配置）→ **拒绝加载并提示**：「配置与已采集数据不匹配，旧数据已失效。请改回配置，或用『新建实验』复制此配置再改维度。」绝不静默把旧数据喂进新模型。
- 加维度的唯一正道：**新建实验文件夹**，而非改旧的。这样"起始点需重做"自动满足——新文件夹本就要重新 `init`。

## 核心库重构

引入两个抽象，替代模块全局变量：

```python
@dataclass(frozen=True)
class Config:
    params: list[Param]
    objectives: list[Objective]
    n_init: int
    seed: int
    batch: int
    num_restarts: int
    raw_samples: int
    mc_samples: int

    @classmethod
    def from_yaml(cls, path) -> "Config": ...
    def fieldnames(self) -> list[str]: ...          # idx,phase,<params>,<objectives>,time
    def fingerprint(self) -> str: ...               # 归一化配置的稳定 hash
    def bounds_tensor(self) -> torch.Tensor: ...

class Experiment:                                    # 绑定一个文件夹
    cfg: Config
    dir: Path
    def load_trials(self) -> list[dict]: ...
    def save_trials(self, rows) -> None: ...
    def is_complete(self, r) -> bool: ...
    def check_conflict(self) -> str | None: ...      # None=OK，否则返回提示文案
    def init_sobol(self) -> None: ...                # 生成起始点 + 写 meta.json
    def to_XY(self): ...
    def fit_model(self, X, Y): ...
    def suggest_next(self) -> torch.Tensor: ...
    def pareto_and_hv(self): ...
```

原 `load_trials / save_trials / is_complete / to_XY / fit_model / ref_point / suggest_next / pareto_and_hv / next_idx` 等从读全局改为读 `cfg` / `exp`。这是有分量的重构，触及 core 里几乎每个函数，但换来配置可测试、可切换、GUI 不再与固定维度焊死。

模块级辅助（仅依赖 `cfg`，不依赖目录）可作为接受 `cfg` 的自由函数，被 `Experiment` 内部调用。

## GUI 改动

- 顶部新增**实验下拉框** + **「新建实验」**按钮。
- 切换下拉 → `_load_experiment(name)`：维度数会变，需**重建**历史表格列、当前待测面板、可视化目标单选钮。现有这些在构造时按 `core.PARAMS` 写死，需抽成可重建方法（如 `_rebuild_for_config()`）。
- **「新建实验」**流程：弹框输入名字 → 建 `experiments/<名字>/` → 复制当前实验的 `config.yaml` 作模板 → 用系统默认编辑器打开（Windows `os.startfile`）→ 提示「编辑好维度后回来点『初始化』」。GUI 内不内嵌 YAML 文本编辑器（YAGNI）。
- 加载实验时先调 `check_conflict()`，有冲突则弹框报错并不加载该实验的数据。
- GUI 持有 `self.exp: Experiment`，所有刷新/绘图/提交改为走 `self.exp` 而非 `core` 全局。

## 现有数据迁移

首次启动若没有 `experiments/` 目录但根目录存在旧 `trials.csv`：
- 自动创建 `experiments/default/`；
- 按当前硬编码的 PARAMS/OBJECTIVES 生成 `config.yaml`；
- 把 `trials.csv` 移入；
- 写 `meta.json`（指纹按生成的 config 计算）；
- 设 `.current = default`。

现有 6 个 init 点不丢。

## CLI 兼容

`fdm_mobo.py` 各子命令新增 `--exp <名字>` 选项（默认读 `experiments/.current`，否则 `default`）。子命令逻辑随核心重构改为通过 `Experiment` 操作。

## 测试策略

纯函数/纯逻辑部分写单元测试，不依赖 GUI、不依赖 BoTorch 拟合：
- `Config.from_yaml` 解析（含缺省值）。
- `Config.fieldnames` 列顺序。
- `Config.fingerprint` 稳定性 + 对范围/维度变化敏感。
- `Experiment.check_conflict`：一致返回 None、改维度/改范围返回提示。
- `load_trials / save_trials` 往返（含 NaN ↔ 空串）。
- 迁移逻辑：给定旧根目录 `trials.csv` 能正确生成 `experiments/default/`。

BO 拟合（`fit_model` / `suggest_next`）可用小数据冒烟测试，不强求数值断言。

## 范围外（YAGNI）

- 不做旧数据向新维度的迁移/补值。
- GUI 内不内嵌 YAML 编辑器。
- 不做实验间结果对比视图。
