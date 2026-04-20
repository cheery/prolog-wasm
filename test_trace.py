"""Tests for trace instrumentation (Phase 5 of PLAN.md).

When LPEmitter(trace=True), each clause firing appends a record of the
form [size, proc_id, clause_idx, *inputs] to a GC array trace buffer.
The module exports __trace_init/__trace_reset/__trace_len/__trace_get
plus the __trace_top global so a host debugger can replay execution.

These tests instantiate the compiled modules via wasmtime's Python
bindings (we need to call multiple exports on one instance — wasmtime
--invoke only runs a single function per invocation).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

import wasmtime
from wasmtime import _ffi as _wt_ffi

from lp_form import *
from lp_parser import parse_lp
from lp_pipeline import lp_compile


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Host harness: instantiate once, then call exports freely.
#
# The wasmtime-py 43.x Config class doesn't expose wasm_gc / function_refs
# via attributes, so we reach into the raw FFI to flip them on. The compiled
# modules use i32 GC arrays for the trace buffer, which requires both flags.
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
        f = self._exports[name]
        return f(self.store, *args)

    def trace(self):
        """Return the raw trace buffer as a list of ints."""
        n = self.call("__trace_len")
        return [self.call("__trace_get", i) for i in range(n)]


def decode_trace(cells):
    """Parse the flat cells into a list of (proc_id, clause_idx, [inputs])."""
    records = []
    i = 0
    while i < len(cells):
        size = cells[i]
        proc_id = cells[i + 1]
        clause_idx = cells[i + 2]
        inputs = cells[i + 3 : i + 1 + size]
        records.append((proc_id, clause_idx, inputs))
        i += 1 + size
    return records


# ---------------------------------------------------------------------------
# Test 1: gcd trace — deterministic clause firings
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


def test_gcd_trace():
    wasm = lp_compile(make_gcd(), trace=True)
    tm = TracedModule(wasm)

    result = tm.call("run", 12, 8)
    assert result == 4, f"gcd(12,8) expected 4, got {result}"

    records = decode_trace(tm.trace())

    # Expected clause firings for gcd(12,8):
    #   gcd(12, 8)  -> clause 0  (b != 0, recurse on (8, 12%8=4))
    #   gcd(8, 4)   -> clause 0  (b != 0, recurse on (4, 8%4=0))
    #   gcd(4, 0)   -> clause 1  (b == 0, return a=4)
    proc_id = 0  # gcd is the only proc -> id 0
    expected = [
        (proc_id, 0, [12, 8]),
        (proc_id, 0, [8, 4]),
        (proc_id, 1, [4, 0]),
    ]
    assert records == expected, (
        f"unexpected trace:\n  got     : {records}\n  expected: {expected}"
    )
    print(f"  test_gcd_trace: {PASS}")


# ---------------------------------------------------------------------------
# Test 2: factorial trace — multi-proc + accumulator
# ---------------------------------------------------------------------------

def test_factorial_trace():
    source = """
        fact(n; ret): fact_acc(n, 1; ret).
        fact_acc(n, acc; ret): n > 0, mul(acc, n; acc'),
                               sub(n, 1; n'), fact_acc(n', acc'; ret).
        fact_acc(n, acc; acc): n == 0.
    """
    prog = parse_lp(source)
    wasm = lp_compile(prog, trace=True)
    tm = TracedModule(wasm)

    result = tm.call("run", 4)
    assert result == 24, f"fact(4) expected 24, got {result}"

    records = decode_trace(tm.trace())

    # Proc ids follow the parse order: fact=0, fact_acc=1.
    # fact(4)           -> fact/clause 0
    # fact_acc(4, 1)    -> fact_acc/clause 0  (n>0; recurse 3,4)
    # fact_acc(3, 4)    -> fact_acc/clause 0  (recurse 2,12)
    # fact_acc(2, 12)   -> fact_acc/clause 0  (recurse 1,24)
    # fact_acc(1, 24)   -> fact_acc/clause 0  (recurse 0,24)
    # fact_acc(0, 24)   -> fact_acc/clause 1
    expected = [
        (0, 0, [4]),
        (1, 0, [4, 1]),
        (1, 0, [3, 4]),
        (1, 0, [2, 12]),
        (1, 0, [1, 24]),
        (1, 1, [0, 24]),
    ]
    assert records == expected, (
        f"unexpected trace:\n  got     : {records}\n  expected: {expected}"
    )
    print(f"  test_factorial_trace: {PASS}")


# ---------------------------------------------------------------------------
# Test 3: run resets trace between invocations
# ---------------------------------------------------------------------------

def test_trace_resets_per_run():
    wasm = lp_compile(make_gcd(), trace=True)
    tm = TracedModule(wasm)

    tm.call("run", 12, 8)
    first_len = tm.call("__trace_len")
    assert first_len > 0

    tm.call("run", 7, 0)           # terminates in one step: clause 1
    second_records = decode_trace(tm.trace())
    assert second_records == [(0, 1, [7, 0])], \
        f"second run should reset trace: {second_records}"
    print(f"  test_trace_resets_per_run: {PASS}")


# ---------------------------------------------------------------------------
# Test 4: manual __trace_reset without rerunning
# ---------------------------------------------------------------------------

def test_trace_reset_export():
    wasm = lp_compile(make_gcd(), trace=True)
    tm = TracedModule(wasm)
    tm.call("run", 12, 8)
    assert tm.call("__trace_len") > 0
    tm.call("__trace_reset")
    assert tm.call("__trace_len") == 0
    print(f"  test_trace_reset_export: {PASS}")


# ---------------------------------------------------------------------------
# Test 5: trace=False leaves the module unchanged (no trace exports)
# ---------------------------------------------------------------------------

def test_trace_flag_optional():
    wasm = lp_compile(make_gcd(), trace=False)
    tm = TracedModule(wasm)
    exports = set(e.name for e in tm.module.exports)
    assert "run" in exports
    assert "__trace_len" not in exports
    assert "__trace_top" not in exports
    assert tm.call("run", 12, 8) == 4
    print(f"  test_trace_flag_optional: {PASS}")


# ---------------------------------------------------------------------------
# Test 6: WAM runtime trace — end-to-end sanity check
# ---------------------------------------------------------------------------

def test_wam_runtime_trace():
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        source = f.read()
    prog = parse_lp(source, entry="test")
    wasm = lp_compile(prog, trace=True)

    tm = TracedModule(wasm)
    result = tm.call("run")
    assert result == 42, f"unify test expected 42, got {result}"

    n = tm.call("__trace_len")
    assert n > 0, "WAM runtime trace should be non-empty"

    # Every record must be well-formed (size > 0, within buffer).
    records = decode_trace(tm.trace())
    assert all(size_ok(rec) for rec in records)

    # The first record should be for the entry procedure `test`.
    proc_names = [p.name for p in prog.procedures]
    test_id = proc_names.index("test")
    assert records[0][0] == test_id, \
        f"first trace record should be `test`: got proc_id={records[0][0]}"

    print(f"  test_wam_runtime_trace: {PASS} ({n} cells, "
          f"{len(records)} records)")


def size_ok(rec):
    proc_id, clause_idx, inputs = rec
    return proc_id >= 0 and clause_idx >= 0 and isinstance(inputs, list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trace instrumentation tests ===")
    tests = [
        test_trace_flag_optional,
        test_gcd_trace,
        test_factorial_trace,
        test_trace_resets_per_run,
        test_trace_reset_export,
        test_wam_runtime_trace,
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
