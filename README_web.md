# FDM MOBO — Web 版

把原 tkinter 桌面 GUI（[fdm_mobo_gui.py](fdm_mobo_gui.py)）封装为浏览器访问的 Web App。
界面布局保持基本不变，并**新增数据下载接口**。复用同一套核心逻辑（[fdm_mobo.py](fdm_mobo.py)）
和数据存储（`experiments/<名称>/trials.csv`），与桌面版完全兼容、可混用。

## 运行

```bash
pip install -r requirements-web.txt      # flask / matplotlib / pyyaml（BO 另需 botorch）
python fdm_mobo_web.py                    # 默认 http://127.0.0.1:5000
# 换端口： PORT=8080 python fdm_mobo_web.py
```

打开浏览器访问提示的地址即可。

## 界面（与桌面版对应）

- **顶部实验栏**：切换实验、新建实验、查看/编辑 `config.yaml`。
- **左侧历史表格**：所有 trial；待测点橙色、Pareto 点蓝色加粗；**双击参数/结果单元格可编辑**。
- **右侧当前待测 + 统计**：大字显示当前待测参数，填写测量结果后提交；统计显示完成数、超体积、各目标最佳值。
- **底部可视化**：参数空间 + 目标空间（Pareto 前沿）双图，由服务端用与桌面版相同的 matplotlib 代码渲染，确保观感一致。

## 新增：数据下载接口

顶部右侧两个按钮，或直接访问以下 URL（`exp` 省略则取当前实验）：

| 接口 | 说明 |
|------|------|
| `GET /api/download/trials?exp=<名称>` | 下载该实验的 `trials.csv` |
| `GET /api/download/config?exp=<名称>` | 下载该实验的 `config.yaml` |
| `GET /api/trials?exp=<名称>` | 结构化 JSON（行数据 + Pareto + 统计），便于程序化抓取 |
| `GET /api/plot.png?exp=<名称>&color=<目标>` | 当前可视化 PNG |

## 说明

- 内置 Flask 开发服务器仅供本机/内网使用；如需对外提供服务，请用 gunicorn/waitress 等 WSGI 服务器部署。
- 桌面版「打开配置」会调用系统编辑器；Web 版改为在浏览器内弹窗编辑并保存（`/api/config/raw`）。
