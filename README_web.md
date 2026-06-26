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

## Linux 一键部署

用 [deploy.sh](deploy.sh)（gunicorn 生产级 WSGI，默认 **5002** 端口）：

```bash
chmod +x deploy.sh

./deploy.sh                    # 建虚拟环境 + 装依赖 + 前台启动 http://0.0.0.0:5002
./deploy.sh --no-bo            # 跳过 botorch(不装 torch,体积小;但无法用「获取 BO 建议」)
PORT=8080 ./deploy.sh          # 改端口

# 装成开机自启的后台常驻服务(systemd)：
./deploy.sh --install-service
sudo systemctl status fdm-mobo-web      # 查看状态
journalctl -u fdm-mobo-web -f           # 查看日志
./deploy.sh --uninstall-service         # 卸载
```

可用环境变量覆盖：`PORT`(5002)、`HOST`(0.0.0.0)、`WORKERS`(2)、`THREADS`(4)、`TIMEOUT`(300s)、`SERVICE_NAME`(fdm-mobo-web)。

> 多 worker 下各实验数据仍共享同一份 `experiments/` 目录，但本应用面向单人 human-in-the-loop 使用，并发写入需自行避免。

## 说明

- 内置 Flask 开发服务器(`python fdm_mobo_web.py`)仅供本机/开发使用；对外提供服务请用上面的 `deploy.sh`(gunicorn)。
- 桌面版「打开配置」会调用系统编辑器；Web 版改为在浏览器内弹窗编辑并保存（`/api/config/raw`）。
