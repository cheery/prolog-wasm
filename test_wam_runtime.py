"""Tests for WAM runtime compiled from LP Form source.

Parses wam_runtime.lp, compiles to WASM, runs with wasmtime.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from lp_parser import parse_lp
from lp_pipeline import lp_compile

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

WASMTIME = "/home/cheery/.wasmtime/bin/wasmtime"


def run_wasm(wasm_bytes, func_name, *args):
    import subprocess, tempfile
    path = tempfile.mktemp(suffix=".wasm")
    with open(path, 'wb') as f:
        f.write(wasm_bytes)
    try:
        cmd = [
            WASMTIME, "-W", "all-proposals=y",
            "--invoke", func_name, path,
        ] + [str(a) for a in args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  RUN ERROR:\n{r.stderr}")
            return None
        line = r.stdout.strip()
        parts = line.split()
        if len(parts) == 1:
            return int(parts[0])
        return tuple(int(p) for p in parts)
    finally:
        os.unlink(path)


def validate_wasm(wasm_bytes, label):
    import subprocess, tempfile
    path = tempfile.mktemp(suffix=".wasm")
    with open(path, 'wb') as f:
        f.write(wasm_bytes)
    try:
        r = subprocess.run(
            [WASMTIME, "compile", "-W", "all-proposals=y", path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  VALIDATE FAIL {label}:\n{r.stderr}")
            return False
        return True
    finally:
        os.unlink(path)
        cwasm = path.replace('.wasm', '.cwasm')
        if os.path.exists(cwasm):
            os.unlink(cwasm)


def compile_runtime():
    """Parse and compile the WAM runtime LP Form source."""
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        source = f.read()
    prog = parse_lp(source, entry="test")
    return lp_compile(prog)


def test_validate():
    wasm = compile_runtime()
    assert validate_wasm(wasm, "wam_runtime"), "WASM validation failed"
    print(f"  test_validate: {PASS}")
    return wasm


def test_unify(wasm):
    """Test: unify f(X) with f(42), expect X = 42."""
    result = run_wasm(wasm, "run")
    assert result == 42, f"expected 42, got {result}"
    print(f"  test_unify: {PASS}")


if __name__ == "__main__":
    print("=== WAM Runtime (LP Form) tests ===")
    passed = 0
    failed = 0
    try:
        wasm = test_validate()
        passed += 1
    except Exception as e:
        import traceback
        print(f"  test_validate: {FAIL} — {e}")
        traceback.print_exc()
        failed += 1
        sys.exit(1)

    tests = [
        lambda: test_unify(wasm),
    ]
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  {FAIL} — {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
