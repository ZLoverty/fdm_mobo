#!/usr/bin/env bash
#
# deploy.sh — FDM MOBO Web 一键部署脚本 (Linux)
#
# 用法:
#   ./deploy.sh                 # 建虚拟环境 + 装依赖 + 用 gunicorn 在 5002 端口启动(前台)
#   ./deploy.sh --no-bo         # 跳过 botorch(不装 torch,体积小,但无法用「获取 BO 建议」)
#   ./deploy.sh --install-service   # 安装为 systemd 服务,开机自启、后台常驻
#   ./deploy.sh --uninstall-service # 卸载 systemd 服务
#   PORT=8080 ./deploy.sh       # 改端口(默认 5002)
#
set -euo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────
PORT="${PORT:-5002}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-300}"          # BO 计算较慢,给足超时
SERVICE_NAME="${SERVICE_NAME:-fdm-mobo-web}"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
GUNICORN="$VENV_DIR/bin/gunicorn"

INSTALL_BO=1
ACTION="run"
for arg in "$@"; do
  case "$arg" in
    --no-bo)             INSTALL_BO=0 ;;
    --install-service)   ACTION="install-service" ;;
    --uninstall-service) ACTION="uninstall-service" ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 1 ;;
  esac
done

# ─── 工具函数 ────────────────────────────────────────────────────────
log() { printf '\033[1;32m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

find_python() {
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  err "未找到 python3,请先安装 Python 3.10+"; exit 1
}

setup_venv() {
  if [[ ! -x "$PY" ]]; then
    local sys_py; sys_py="$(find_python)"
    log "创建虚拟环境 ($sys_py) -> $VENV_DIR"
    "$sys_py" -m venv "$VENV_DIR"
  else
    log "复用已有虚拟环境 $VENV_DIR"
  fi
  log "升级 pip"
  "$PY" -m pip install --quiet --upgrade pip
  log "安装 Web 依赖 (flask / pyyaml / gunicorn)"
  "$PIP" install --quiet flask pyyaml gunicorn
  if [[ "$INSTALL_BO" -eq 1 ]]; then
    if "$PY" -c "import botorch" >/dev/null 2>&1; then
      log "botorch 已安装,跳过"
    else
      log "安装 botorch (含 torch,体积较大,请耐心等待)…"
      "$PIP" install --quiet botorch
    fi
  else
    log "已跳过 botorch (--no-bo):BO 建议功能将不可用"
  fi
}

run_foreground() {
  log "启动 gunicorn  http://$HOST:$PORT  (workers=$WORKERS threads=$THREADS)"
  log "按 Ctrl+C 停止"
  cd "$APP_DIR"
  exec "$GUNICORN" \
    --chdir "$APP_DIR" \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --threads "$THREADS" \
    --timeout "$TIMEOUT" \
    --access-logfile - \
    fdm_mobo_web:app
}

install_service() {
  if ! command -v systemctl >/dev/null 2>&1; then
    err "本系统无 systemd,无法安装服务。可改用前台模式 ./deploy.sh"; exit 1
  fi
  local run_user="${SUDO_USER:-$USER}"
  local unit="/etc/systemd/system/${SERVICE_NAME}.service"
  log "写入 systemd 单元 $unit (运行用户: $run_user)"
  sudo tee "$unit" >/dev/null <<EOF
[Unit]
Description=FDM MOBO Web (Bayesian optimization GUI)
After=network.target

[Service]
Type=simple
User=$run_user
WorkingDirectory=$APP_DIR
ExecStart=$GUNICORN --chdir $APP_DIR --bind $HOST:$PORT --workers $WORKERS --threads $THREADS --timeout $TIMEOUT fdm_mobo_web:app
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl restart "$SERVICE_NAME"
  log "服务已启动并设为开机自启。"
  log "  状态:  sudo systemctl status $SERVICE_NAME"
  log "  日志:  journalctl -u $SERVICE_NAME -f"
  log "  访问:  http://<本机IP>:$PORT"
}

uninstall_service() {
  log "停止并卸载服务 $SERVICE_NAME"
  sudo systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
  sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
  sudo systemctl daemon-reload
  log "已卸载。"
}

# ─── 主流程 ──────────────────────────────────────────────────────────
case "$ACTION" in
  uninstall-service) uninstall_service ;;
  install-service)   setup_venv; install_service ;;
  run)               setup_venv; run_foreground ;;
esac
