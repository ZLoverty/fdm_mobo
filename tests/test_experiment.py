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


def test_load_trials_empty_when_missing(tmp_path):
    exp = _make_exp(tmp_path)
    assert exp.load_trials() == []


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
