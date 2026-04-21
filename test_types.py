"""Tests for the LP Form type system (Phase 9).

Covers the three layers described in DT-PLAN.md:

  Layer 1 — structs.       Tested end-to-end (parse, compile, run).
  Layer 2 — sum types.     Pattern dispatch, exhaustiveness, CHC emission.
  Layer 3 — ADTs.          Declaration + signature validation.

Sum-type pattern dispatch across procedure boundaries is not end-to-end
tested because it requires cross-procedure ref-typed outputs, which the
current emitter doesn't model (all procedure outputs are i32). The CHC
extractor does see the elaborated form, so the pattern-matching
semantics are exercised there.
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from lp_form import (
    LPProgram, pretty_print, validate,
    LPStructDecl, LPSumDecl, LPConstructor, LPADT, LPSignature,
)
from lp_parser import parse_lp
from lp_pipeline import lp_compile
from lp_elaborate import elaborate, TypeEnv, check_exhaustive_dispatch
from chc import extract_chc


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WASMTIME = "wasmtime"


def _run_wasm(wasm_bytes, func_name, *args):
    path = tempfile.mktemp(suffix=".wasm")
    with open(path, "wb") as f:
        f.write(wasm_bytes)
    try:
        r = subprocess.run(
            [WASMTIME, "-W", "all-proposals=y", "--invoke", func_name,
             path] + [str(a) for a in args],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise AssertionError(f"wasmtime failed: {r.stderr[:400]}")
        parts = r.stdout.strip().split()
        if len(parts) == 1:
            return int(parts[0])
        return tuple(int(p) for p in parts)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Layer 1 — structs
# ---------------------------------------------------------------------------

def test_struct_declaration_parses():
    src = """
    struct Cell { tag: i32, val: i32 }.

    id(x; y): copy(x; y).
    """
    prog = parse_lp(src)
    assert len(prog.structs) == 1
    s = prog.structs[0]
    assert s.name == "Cell"
    assert s.fields == [("tag", "i32"), ("val", "i32")]
    print(f"  test_struct_declaration_parses: {PASS}")


def test_struct_new_and_field_access():
    src = """
    struct Cell { tag: i32, val: i32 }.

    tag_of(t, v; out):
        struct_new(Cell, t, v; c),
        copy(c.tag; out).
    """
    prog = parse_lp(src)
    wasm = lp_compile(prog)
    assert _run_wasm(wasm, "run", 7, 42) == 7
    assert _run_wasm(wasm, "run", 99, 1) == 99
    print(f"  test_struct_new_and_field_access: {PASS}")


def test_struct_second_field():
    src = """
    struct Cell { tag: i32, val: i32 }.

    val_of(t, v; out):
        struct_new(Cell, t, v; c),
        copy(c.val; out).
    """
    prog = parse_lp(src)
    wasm = lp_compile(prog)
    assert _run_wasm(wasm, "run", 7, 42) == 42
    assert _run_wasm(wasm, "run", 99, 1) == 1
    print(f"  test_struct_second_field: {PASS}")


def test_struct_nested_fields():
    src = """
    struct Point { x: i32, y: i32 }.

    dist_sq(x, y; out):
        struct_new(Point, x, y; p),
        mul(p.x, p.x; xx),
        mul(p.y, p.y; yy),
        add(xx, yy; out).
    """
    prog = parse_lp(src)
    wasm = lp_compile(prog)
    assert _run_wasm(wasm, "run", 3, 4) == 25
    assert _run_wasm(wasm, "run", 5, 12) == 169
    print(f"  test_struct_nested_fields: {PASS}")


def test_struct_chc_axioms():
    src = """
    struct Cell { tag: i32, val: i32 }.

    get_tag(t, v; out):
        struct_new(Cell, t, v; c),
        copy(c.tag; out).
    """
    prog = parse_lp(src)
    smt = extract_chc(prog)
    # Constructor function and accessors are declared.
    assert "(declare-fun Cell (Int Int) Int)" in smt
    assert "(declare-fun Cell_tag (Int) Int)" in smt
    assert "(declare-fun Cell_val (Int) Int)" in smt
    # Axiom: Cell_tag (Cell t v) = t
    assert "(Cell_tag (Cell v0 v1)) v0" in smt
    assert "(Cell_val (Cell v0 v1)) v1" in smt
    print(f"  test_struct_chc_axioms: {PASS}")


# ---------------------------------------------------------------------------
# Layer 2 — sum types and pattern matching
# ---------------------------------------------------------------------------

def test_sum_type_parses():
    src = """
    type Cell = ref(i32) | con(i32) | fun(i32, i32).

    id(x; y): copy(x; y).
    """
    prog = parse_lp(src)
    assert len(prog.sums) == 1
    s = prog.sums[0]
    assert s.name == "Cell"
    assert [c.name for c in s.constructors] == ["ref", "con", "fun"]
    assert s.constructors[0].params == ["i32"]
    assert s.constructors[2].params == ["i32", "i32"]
    print(f"  test_sum_type_parses: {PASS}")


def test_pattern_dispatch_elaborates():
    """Elaboration turns pattern dispatch into tag-guards + struct_get."""
    src = """
    type Cell = ref(i32) | con(i32) | fun(i32, i32).

    is_ref(a; out):
        source(a; ref(t)), copy(1; out).
    is_ref(a; out):
        source(a; con(v)), copy(0; out).
    is_ref(a; out):
        source(a; fun(x, y)), copy(0; out).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    elab = elaborate(prog)
    # First clause should have Call, struct_get(__tag), Guard(tag==0),
    # struct_get(_f0), copy.
    clause0 = elab.procedures[0].clauses[0]
    ops = [g.op if hasattr(g, 'op') else type(g).__name__
           for g in clause0.goals]
    assert "struct_get" in ops
    print(f"  test_pattern_dispatch_elaborates: {PASS}")


def test_exhaustive_dispatch_passes():
    src = """
    type Cell = ref(i32) | con(i32) | fun(i32, i32).

    kind(a; k):
        source(a; ref(t)), copy(1; k).
    kind(a; k):
        source(a; con(v)), copy(2; k).
    kind(a; k):
        source(a; fun(x, y)), copy(3; k).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    elaborate(prog)  # should not raise
    print(f"  test_exhaustive_dispatch_passes: {PASS}")


def test_non_exhaustive_dispatch_rejected():
    src = """
    type Cell = ref(i32) | con(i32) | fun(i32, i32).

    kind(a; k):
        source(a; ref(t)), copy(1; k).
    kind(a; k):
        source(a; con(v)), copy(2; k).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    try:
        elaborate(prog)
    except ValueError as e:
        assert "non-exhaustive" in str(e)
        assert "fun" in str(e)
        print(f"  test_non_exhaustive_dispatch_rejected: {PASS}")
        return
    raise AssertionError("expected ValueError for non-exhaustive dispatch")


def test_wildcard_in_dispatch_rejected():
    src = """
    type Cell = ref(i32) | con(i32).

    deref(a; r):
        source(a; ref(t)), copy(t; r).
    deref(a; r):
        source(a; _()), copy(a; r).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    try:
        elaborate(prog)
    except ValueError as e:
        assert "wildcard" in str(e)
        print(f"  test_wildcard_in_dispatch_rejected: {PASS}")
        return
    raise AssertionError("expected ValueError for wildcard dispatch")


def test_mixed_dispatch_rejected():
    """A procedure may not mix pattern clauses with plain-variable clauses."""
    src = """
    type Cell = ref(i32) | con(i32).

    deref(a; r):
        source(a; ref(t)), copy(t; r).
    deref(a; r):
        source(a; c), copy(a; r).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    try:
        elaborate(prog)
    except ValueError as e:
        assert "mixes pattern dispatch" in str(e)
        print(f"  test_mixed_dispatch_rejected: {PASS}")
        return
    raise AssertionError("expected ValueError for mixed dispatch")


def test_sum_chc_emission():
    src = """
    type Cell = ref(i32) | con(i32) | fun(i32, i32).

    kind(a; k):
        source(a; ref(t)), copy(1; k).
    kind(a; k):
        source(a; con(v)), copy(2; k).
    kind(a; k):
        source(a; fun(x, y)), copy(3; k).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    smt = extract_chc(prog)
    # Tag accessor and payload accessors declared.
    assert "(declare-fun Cell___tag (Int) Int)" in smt
    assert "(declare-fun Cell__f0 (Int) Int)" in smt
    assert "(declare-fun Cell__f1 (Int) Int)" in smt
    # Each clause's guard uses the tag accessor on a handle.
    assert "Cell___tag" in smt
    # Parses in z3.
    import z3
    s = z3.SolverFor("HORN")
    s.from_string(smt)
    print(f"  test_sum_chc_emission: {PASS}")


def test_sum_constructor_with_args_elaborates():
    """Two-argument constructor binds both payload vars."""
    src = """
    type Cell = unary(i32) | pair(i32, i32).

    first(a; out):
        source(a; unary(v)), copy(v; out).
    first(a; out):
        source(a; pair(x, y)), copy(x; out).

    source(a; r): copy(a; r).
    """
    prog = parse_lp(src)
    elab = elaborate(prog)
    # Second clause should have two struct_get ops (one per payload).
    clause1 = elab.procedures[0].clauses[1]
    sget = [g for g in clause1.goals
            if hasattr(g, 'op') and g.op == "struct_get"]
    # One for tag, two for payload fields.
    assert len(sget) == 3
    print(f"  test_sum_constructor_with_args_elaborates: {PASS}")


# ---------------------------------------------------------------------------
# Layer 3 — ADTs
# ---------------------------------------------------------------------------

def test_adt_declaration_parses():
    src = """
    adt Counter {
        init(n; c)
        inc(c; c_new)
        get(c; v)
    }.

    init(n; c): copy(n; c).
    inc(c; c_new): add(c, 1; c_new).
    get(c; v): copy(c; v).
    """
    prog = parse_lp(src)
    assert len(prog.adts) == 1
    a = prog.adts[0]
    assert a.name == "Counter"
    names = [s.name for s in a.signatures]
    assert names == ["init", "inc", "get"]
    assert all(s.arity_in == 1 and s.arity_out == 1
               for s in a.signatures)
    print(f"  test_adt_declaration_parses: {PASS}")


def test_adt_validation_missing_impl():
    src = """
    adt Counter {
        init(n; c)
        inc(c; c_new)
    }.

    init(n; c): copy(n; c).
    """
    prog = parse_lp(src)
    try:
        validate(prog)
    except ValueError as e:
        assert "inc" in str(e)
        assert "no implementing procedure" in str(e)
        print(f"  test_adt_validation_missing_impl: {PASS}")
        return
    raise AssertionError("expected validation error for missing impl")


def test_adt_validation_arity_mismatch():
    src = """
    adt Counter {
        inc(c, step; c_new)
    }.

    inc(c; c_new): add(c, 1; c_new).
    """
    prog = parse_lp(src)
    try:
        validate(prog)
    except ValueError as e:
        assert "arity" in str(e)
        print(f"  test_adt_validation_arity_mismatch: {PASS}")
        return
    raise AssertionError("expected validation error for arity mismatch")


def test_adt_pretty_print():
    src = """
    adt Stack {
        push(s, v; s_new)
        pop(s; s_new, v)
    }.

    push(s, v; s_new): add(s, v; s_new).
    pop(s; s_new, v): copy(s; s_new), copy(s; v).
    """
    prog = parse_lp(src)
    rendered = pretty_print(prog)
    assert "adt Stack {" in rendered
    assert "push" in rendered
    assert "pop" in rendered
    print(f"  test_adt_pretty_print: {PASS}")


def test_adt_compiles_to_wasm():
    """ADT declarations don't affect code generation — procs compile as usual."""
    src = """
    adt Counter {
        init(n; c)
        inc(c; c_new)
    }.

    init(n; c): copy(n; c).
    inc(c; c_new): add(c, 1; c_new).
    """
    prog = parse_lp(src)
    prog.entry = "inc"
    wasm = lp_compile(prog)
    assert _run_wasm(wasm, "run", 5) == 6
    print(f"  test_adt_compiles_to_wasm: {PASS}")


# ---------------------------------------------------------------------------
# Regression: pre-existing programs still parse
# ---------------------------------------------------------------------------

def test_wam_runtime_still_parses():
    with open(os.path.join(os.path.dirname(__file__),
                           'wam_runtime.lp')) as f:
        prog = parse_lp(f.read())
    validate(prog)
    print(f"  test_wam_runtime_still_parses: {PASS}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_struct_declaration_parses,
    test_struct_new_and_field_access,
    test_struct_second_field,
    test_struct_nested_fields,
    test_struct_chc_axioms,
    test_sum_type_parses,
    test_pattern_dispatch_elaborates,
    test_exhaustive_dispatch_passes,
    test_non_exhaustive_dispatch_rejected,
    test_wildcard_in_dispatch_rejected,
    test_mixed_dispatch_rejected,
    test_sum_chc_emission,
    test_sum_constructor_with_args_elaborates,
    test_adt_declaration_parses,
    test_adt_validation_missing_impl,
    test_adt_validation_arity_mismatch,
    test_adt_pretty_print,
    test_adt_compiles_to_wasm,
    test_wam_runtime_still_parses,
]


if __name__ == "__main__":
    print("=== LP Form type system tests ===")
    passed = 0
    failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  {t.__name__}: {FAIL} — {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
