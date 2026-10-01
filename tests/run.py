"""Minimal test runner (no pytest dependency): python -m tests.run"""
import importlib, inspect, sys, traceback
mods = ["tests.test_sim", "tests.test_data"]
failed = 0
for m in mods:
    mod = importlib.import_module(m)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            try:
                fn(); print("ok  ", name)
            except Exception:
                failed += 1; print("FAIL", name); traceback.print_exc()
sys.exit(1 if failed else 0)
