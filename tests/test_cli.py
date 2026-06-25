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
