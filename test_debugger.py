"""Tests for the Phase 7 trace-driven debugger.

Covers:
  - forward replay of a trace (gcd, wam_runtime)
  - backward step via snapshots
  - goto with and without snapshots (must agree)
  - invertibility-gated trace elision (leaf pure proc)
  - emitter rejection of invertible procs with Calls or mutations
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

import wasmtime
from wasmtime import _ffi as _wt_ffi

from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst,
)
from lp_parser import parse_lp
from lp_pipeline import lp_compile
from trace_tool import Trace, Replay, TraceRecord, Frame


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Shared wasmtime harness
# ---------------------------------------------------------------------------

def make_engine():
    cfg = wasmtime.Config()
    _wt_ffi.wasmtime_config_wasm_gc_set(cfg.ptr(), True)
    _wt_ffi.wasmtime_config_wasm_function_references_set(cfg.ptr(), True)
    cfg.wasm_reference_types = True
    cfg.wasm_tail_call = True
    cfg.wasm_multi_value = True
    return wasmtime.Engine(cfg)


class TracedModule:
    def __init__(self, wasm_bytes):
        self.engine = make_engine()
        self.store = wasmtime.Store(self.engine)
        self.module = wasmtime.Module(self.engine, wasm_bytes)
        self.instance = wasmtime.Instance(self.store, self.module, [])
        self._exports = {
            e.name: self.instance.exports(self.store)[e.name]
            for e in self.module.exports
        }

    def call(self, name, *args):
        return self._exports[name](self.store, *args)

    def trace(self):
        n = self.call("__trace_len")
        return [self.call("__trace_get", i) for i in range(n)]


def run_and_trace(prog, *entry_args):
    wasm = lp_compile(prog, trace=True)
    tm = TracedModule(wasm)
    result = tm.call("run", *entry_args)
    return result, tm.trace(), tm


# ---------------------------------------------------------------------------
# Fixture programs
# ---------------------------------------------------------------------------

def make_gcd():
    return LPProgram(
        procedures=[
            LPProc("gcd", 2, 1, [
                LPClause(
                    head=LPHead("gcd", ["a", "b"], ["ret"]),
                    goals=[
                        Guard("ne", LPVar("b"), LPConst(0)),
                        PrimOp("rem", [LPVar("a"), LPVar("b")], ["b_prime"]),
                        Call("gcd", [LPVar("b"), LPVar("b_prime")], ["ret"]),
                    ],
                ),
                LPClause(
                    head=LPHead("gcd", ["a", "b"], ["ret"]),
                    goals=[
                        Guard("eq", LPVar("b"), LPConst(0)),
                        PrimOp("copy", [LPVar("a")], ["ret"]),
                    ],
                ),
            ]),
        ],
        entry="gcd",
    )


def make_add3():
    """main(x, y, z; ret) calls add_pair twice. add_pair is a pure leaf."""
    return LPProgram(
        procedures=[
            LPProc("main", 3, 1, [
                LPClause(
                    head=LPHead("main", ["x", "y", "z"], ["ret"]),
                    goals=[
                        Call("add_pair", [LPVar("x"), LPVar("y")], ["t"]),
                        Call("add_pair", [LPVar("t"), LPVar("z")], ["ret"]),
                    ],
                ),
            ]),
            LPProc("add_pair", 2, 1, [
                LPClause(
                    head=LPHead("add_pair", ["a", "b"], ["r"]),
                    goals=[
                        PrimOp("add", [LPVar("a"), LPVar("b")], ["r"]),
                    ],
                ),
            ]),
        ],
        entry="main",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_replay_gcd():
    prog = make_gcd()
    result, cells, _ = run_and_trace(prog, 12, 8)
    assert result == 4

    trace = Trace(cells, prog)
    replay = Replay(prog, trace)
    while replay.cursor < len(trace):
        replay.step()

    assert replay.final_outputs is not None, "final_outputs never set"
    assert replay.final_outputs["ret"] == 4, \
        f"expected ret=4, got {replay.final_outputs}"
    assert replay.frames == [], "stack should be empty at end"
    print(f"  test_forward_replay_gcd: {PASS}")


def test_forward_replay_wam_runtime():
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        source = f.read()
    prog = parse_lp(source, entry="test")

    result, cells, tm = run_and_trace(prog)
    assert result == 42

    trace = Trace(cells, prog)
    replay = Replay(prog, trace)
    while replay.cursor < len(trace):
        replay.step()

    assert replay.final_outputs is not None
    assert replay.final_outputs["result"] == 42, \
        f"expected result=42, got {replay.final_outputs}"
    # Sanity: HEAP should have been allocated and populated.
    assert "HEAP" in replay.arrays
    assert replay.arrays["HEAP"][0] == 4  # tag=FUN at cell 0
    assert replay.globals["FAIL"] == 0
    print(f"  test_forward_replay_wam_runtime: {PASS} "
          f"({len(trace)} records replayed)")


def test_back_step():
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        source = f.read()
    prog = parse_lp(source, entry="test")
    _, cells, _ = run_and_trace(prog)
    trace = Trace(cells, prog)

    replay = Replay(prog, trace, snapshot_interval=5)
    # Step to some midpoint
    K = min(20, len(trace) - 5)
    replay.goto(K)

    # Snapshot observed state at K
    before_H = replay.globals.get("H", 0)
    before_TR = replay.globals.get("TR", 0)
    heap_slice = list(replay.arrays["HEAP"][:20]) \
        if "HEAP" in replay.arrays else []

    # Advance forward 5, then back 5
    replay.goto(K + 5)
    for _ in range(5):
        replay.back()

    assert replay.cursor == K, f"cursor {replay.cursor} != {K}"
    assert replay.globals.get("H", 0) == before_H
    assert replay.globals.get("TR", 0) == before_TR
    after_slice = list(replay.arrays["HEAP"][:20]) \
        if "HEAP" in replay.arrays else []
    assert after_slice == heap_slice, "HEAP prefix diverged after back()"
    print(f"  test_back_step: {PASS} (K={K})")


def test_goto_matches_with_and_without_snapshots():
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        source = f.read()
    prog = parse_lp(source, entry="test")
    _, cells, _ = run_and_trace(prog)
    trace = Trace(cells, prog)

    fast = Replay(prog, trace, snapshot_interval=7)
    slow = Replay(prog, trace, snapshot_interval=0)

    targets = [0, 3, 8, 15, len(trace) // 2, len(trace)]
    for t in targets:
        fast.goto(t)
        slow.goto(t)
        assert fast.globals == slow.globals, \
            f"globals diverge at goto({t}): fast={fast.globals} slow={slow.globals}"
        assert set(fast.arrays) == set(slow.arrays), \
            f"array set diverges at goto({t})"
        for k in fast.arrays:
            assert fast.arrays[k] == slow.arrays[k], \
                f"array {k} diverges at goto({t})"
        assert fast.cursor == slow.cursor == t
    print(f"  test_goto_matches_with_and_without_snapshots: {PASS}")


def test_invertibility_elides_records():
    # Baseline: add_pair traces normally.
    prog_base = make_add3()
    _, cells_base, _ = run_and_trace(prog_base, 1, 2, 3)
    trace_base = Trace(cells_base, prog_base)

    # Elided: mark add_pair invertible.
    prog_elide = make_add3()
    prog_elide.procedures[1].invertible = True
    result, cells_elide, _ = run_and_trace(prog_elide, 1, 2, 3)
    trace_elide = Trace(cells_elide, prog_elide)

    assert result == 6, f"add3(1,2,3) expected 6, got {result}"

    # Baseline has 1 main + 2 add_pair firings = 3 records.
    # Elided has only the main firing = 1 record.
    base_procs = [r.proc_id for r in trace_base]
    elide_procs = [r.proc_id for r in trace_elide]
    add_pair_id = prog_base.procedures[1].name
    main_id = prog_base.procedures[0].name

    assert len(trace_base) == 3, f"baseline should have 3 records, got {len(trace_base)}"
    assert len(trace_elide) == 1, f"elided should have 1 record, got {len(trace_elide)}"
    assert trace_elide[0].proc_id == 0  # main still traces
    # The remaining record in the elided trace is a subsequence of the baseline.
    assert trace_elide[0].proc_id == trace_base[0].proc_id
    assert trace_elide[0].inputs == trace_base[0].inputs
    print(f"  test_invertibility_elides_records: {PASS} "
          f"(3 -> 1 records)")


def test_invertible_with_call_rejected():
    prog = make_add3()
    prog.procedures[0].invertible = True  # main has Calls
    try:
        lp_compile(prog, trace=True)
    except ValueError as e:
        assert "main" in str(e) and "Call" in str(e)
        print(f"  test_invertible_with_call_rejected: {PASS}")
        return
    raise AssertionError("expected ValueError for invertible proc with Call")


def test_invertible_with_mutation_rejected():
    prog = LPProgram(
        procedures=[
            LPProc("set_h", 1, 0, [
                LPClause(
                    head=LPHead("set_h", ["v"], []),
                    goals=[PrimOp("gset", [LPVar("H"), LPVar("v")], [])],
                ),
            ], invertible=True),
        ],
        globals=[__import__('lp_form').GlobalDecl("H", 0)],
        entry="set_h",
    )
    try:
        lp_compile(prog, trace=True)
    except ValueError as e:
        assert "set_h" in str(e) and "gset" in str(e)
        print(f"  test_invertible_with_mutation_rejected: {PASS}")
        return
    raise AssertionError("expected ValueError for invertible proc with gset")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trace-driven debugger tests ===")
    tests = [
        test_forward_replay_gcd,
        test_forward_replay_wam_runtime,
        test_back_step,
        test_goto_matches_with_and_without_snapshots,
        test_invertibility_elides_records,
        test_invertible_with_call_rejected,
        test_invertible_with_mutation_rejected,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  {t.__name__}: {FAIL} — {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
