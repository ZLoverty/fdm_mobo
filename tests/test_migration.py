# tests/test_migration.py
from pathlib import Path
import fdm_mobo as core


def test_migrate_legacy_creates_default(tmp_path):
    legacy = tmp_path / "trials.csv"
    legacy.write_text(
        "idx,phase,fan,flow,surface,TS,time\n"
        "0,init,20,1,,,\n"
        "1,init,50,1,7,30,2026-06-25T00:00:00\n",
        encoding="utf-8",
    )
    root = tmp_path / "experiments"
    name = core.migrate_legacy(legacy_csv=legacy, root=root)
    assert name == "default"
    d = root / "default"
    assert (d / "config.yaml").exists()
    assert (d / "trials.csv").exists()
    assert (d / "meta.json").exists()
    assert not legacy.exists()                       # 已移动
    assert core.get_current(root) == "default"
    exp = core.Experiment(d)
    assert exp.check_conflict() is None              # meta 指纹与生成 config 一致
    rows = exp.load_trials()
    assert len(rows) == 2 and rows[1]["surface"] == 7.0


def test_migrate_noop_when_root_exists(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()
    legacy = tmp_path / "trials.csv"
    legacy.write_text("idx,phase,fan,flow,surface,TS,time\n", encoding="utf-8")
    assert core.migrate_legacy(legacy_csv=legacy, root=root) is None


def test_list_and_current(tmp_path):
    root = tmp_path / "experiments"
    core.create_experiment("a", core.LEGACY_CONFIG_YAML, root=root)
    core.create_experiment("b", core.LEGACY_CONFIG_YAML, root=root)
    assert core.list_experiments(root) == ["a", "b"]
    core.set_current("b", root=root)
    assert core.get_current(root) == "b"


def test_create_experiment_rejects_bad_names(tmp_path):
    import pytest
    root = tmp_path / "experiments"
    for bad in ["", "  ", "a/b", "a\\b", "..", "."]:
        with pytest.raises(ValueError):
            core.create_experiment(bad, core.LEGACY_CONFIG_YAML, root=root)


def test_migrate_noop_when_csv_missing(tmp_path):
    root = tmp_path / "experiments"
    missing = tmp_path / "nope.csv"
    assert core.migrate_legacy(legacy_csv=missing, root=root) is None


def test_get_current_none_on_empty_file(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()
    (root / ".current").write_text("   ", encoding="utf-8")
    assert core.get_current(root) is None
