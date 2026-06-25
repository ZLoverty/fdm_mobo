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


def test_bounds_tensor_shape_and_values(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    cfg = fc.Config.from_yaml(_write(tmp_path, BASE))
    b = cfg.bounds_tensor()
    assert b.shape == (2, 2)
    assert b.dtype == torch.double
    assert b[0].tolist() == [0.0, 0.9]
    assert b[1].tolist() == [100.0, 1.1]


def test_rejects_invalid_goal(tmp_path):
    import pytest
    text = """\
params:
  - {name: fan, low: 0, high: 100}
objectives:
  - {name: TS, goal: maximize}
"""
    with pytest.raises(ValueError):
        fc.Config.from_yaml(_write(tmp_path, text))
