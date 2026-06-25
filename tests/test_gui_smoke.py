# tests/test_gui_smoke.py
import importlib


def test_gui_module_imports():
    # 仅验证模块语法与符号存在，不实例化 Tk（无显示环境）
    mod = importlib.import_module("fdm_mobo_gui")
    assert hasattr(mod, "App")
