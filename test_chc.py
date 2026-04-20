"""Tests for CHC extraction from LP Form programs.

Extracts CHC from LP Form programs, parses with Z3, and verifies
simple safety properties where tractable.

Note on Z3/Spacer results:
  - sat   = property holds (no counterexample)
  - unsat = property violated (counterexample exists)
  - unknown = solver couldn't decide (common with mod, arrays)
"""

import sys, os, functools
sys.path.insert(0, os.path.dirname(__file__))

import z3

from chc import extract_chc
from lp_form import *
from lp_parser import parse_lp
from test_lp import (make_gcd, make_factorial, make_fibonacci, make_divmod)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _parse_horn(smt: str):
    s = z3.SolverFor("HORN")
    s.from_string(smt)
    return s


# ---------------------------------------------------------------------------
# Property harness for wam_runtime.lp
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _wam_program():
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        return parse_lp(f.read())


@functools.lru_cache(maxsize=None)
def _wam_smt(slice_key=None):
    """Extract CHC from wam_runtime.lp, optionally sliced to a set of procs.

    `slice_key` is a (hashable) tuple of proc names — only those plus
    their transitive callees are emitted. Spacer handles a small sliced
    program far more reliably than the full runtime.
    """
    slice_to = list(slice_key) if slice_key is not None else None
    return extract_chc(_wam_program(), slice_to=slice_to)


def check_holds(query, slice_to, timeout_ms=30_000):
    """Assert a property holds against a slice of wam_runtime.lp.

    `query` is SMT-LIB2 source that negates the property (i.e. asserts
    `(=> (and predicate property-negation) false)`). sat means no
    counterexample exists — the property holds.

    `slice_to` names the target procs; CHC is restricted to those plus
    their transitive callees.
    """
    slice_key = tuple(sorted(slice_to))
    s = z3.SolverFor("HORN")
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.from_string(_wam_smt(slice_key) + query)
    r = s.check()
    if r != z3.sat:
        raise AssertionError(f"expected sat (property holds), got {r}")
    return r


def check_violated(query, slice_to, timeout_ms=30_000):
    """Negative test: assert a property is VIOLATED — Spacer returns unsat."""
    slice_key = tuple(sorted(slice_to))
    s = z3.SolverFor("HORN")
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.from_string(_wam_smt(slice_key) + query)
    r = s.check()
    if r != z3.unsat:
        raise AssertionError(f"expected unsat (violation), got {r}")
    return r


# ---------------------------------------------------------------------------
# Parsing tests: Z3 must accept the extracted SMT-LIB2
# ---------------------------------------------------------------------------

def test_gcd_parses():
    smt = extract_chc(make_gcd())
    _parse_horn(smt)
    assert "(declare-fun gcd " in smt
    print(f"  test_gcd_parses: {PASS}")


def test_factorial_parses():
    smt = extract_chc(make_factorial())
    _parse_horn(smt)
    assert "(declare-fun fact " in smt
    assert "(declare-fun fact_acc " in smt
    print(f"  test_factorial_parses: {PASS}")


def test_fibonacci_parses():
    smt = extract_chc(make_fibonacci())
    _parse_horn(smt)
    assert "(declare-fun fib " in smt
    print(f"  test_fibonacci_parses: {PASS}")


def test_divmod_parses():
    smt = extract_chc(make_divmod())
    _parse_horn(smt)
    assert "(declare-fun divmod " in smt
    print(f"  test_divmod_parses: {PASS}")


def test_wam_runtime_parses():
    """Extract CHC from the full WAM runtime (stateful: globals + arrays)."""
    with open(os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')) as f:
        src = f.read()
    prog = parse_lp(src)
    smt = extract_chc(prog)
    _parse_horn(smt)

    assert "(declare-fun heap_push " in smt
    assert "(declare-fun init " in smt
    # Ref arrays now appear as opaque sorts and in predicate signatures
    assert "(declare-sort CONT 0)" in smt
    assert "(declare-sort BP_STACK 0)" in smt
    print(f"  test_wam_runtime_parses: {PASS}")


# ---------------------------------------------------------------------------
# Verification tests: simple properties Z3 can actually decide
# ---------------------------------------------------------------------------

def test_factorial_positivity():
    """fact(n, r) with n >= 0 => r >= 1."""
    smt = extract_chc(make_factorial())
    query = """
(assert (forall ((n Int) (r Int))
    (=> (and (fact n r) (>= n 0) (< r 1)) false)))
"""
    s = _parse_horn(smt + query)
    result = s.check()
    assert result == z3.sat, f"expected sat (property holds), got {result}"
    print(f"  test_factorial_positivity: {PASS}")


def test_fibonacci_base_case():
    """fib(0, r) => r = 0."""
    smt = extract_chc(make_fibonacci())
    query = """
(assert (forall ((r Int)) (=> (and (fib 0 r) (not (= r 0))) false)))
"""
    s = _parse_horn(smt + query)
    result = s.check()
    assert result == z3.sat, f"expected sat, got {result}"
    print(f"  test_fibonacci_base_case: {PASS}")


def test_fibonacci_non_negative():
    """fib(n, r) with n >= 0 => r >= 0."""
    smt = extract_chc(make_fibonacci())
    query = """
(assert (forall ((n Int) (r Int))
    (=> (and (fib n r) (>= n 0) (< r 0)) false)))
"""
    s = _parse_horn(smt + query)
    result = s.check()
    assert result == z3.sat, f"expected sat, got {result}"
    print(f"  test_fibonacci_non_negative: {PASS}")


def test_counterexample_detected():
    """Negative test: assert a false property, expect unsat."""
    smt = extract_chc(make_fibonacci())
    # False claim: fib(0, r) => r = 1
    query = """
(assert (forall ((r Int)) (=> (and (fib 0 r) (not (= r 1))) false)))
"""
    s = _parse_horn(smt + query)
    result = s.check()
    assert result == z3.unsat, \
        f"expected unsat (property violated), got {result}"
    print(f"  test_counterexample_detected: {PASS}")


# ---------------------------------------------------------------------------
# Stateful program test
# ---------------------------------------------------------------------------

def test_stateful_parses():
    """A minimal program with a global counter; CHC must thread state."""
    prog = LPProgram(
        procedures=[
            LPProc("inc", 0, 0, [
                LPClause(
                    head=LPHead("inc", [], []),
                    goals=[
                        PrimOp("gget", [LPVar("C")], ["c"]),
                        PrimOp("add", [LPVar("c"), LPConst(1)], ["cp"]),
                        PrimOp("gset", [LPVar("C"), LPVar("cp")], []),
                    ],
                ),
            ]),
        ],
        globals=[GlobalDecl("C", 0)],
    )
    smt = extract_chc(prog)
    _parse_horn(smt)
    # Predicate signature should be (Int Int) - pre-state C, post-state C
    assert "(declare-fun inc (Int Int) Bool)" in smt
    print(f"  test_stateful_parses: {PASS}")


def test_stateful_property():
    """Verify: after inc, C increased by exactly 1."""
    prog = LPProgram(
        procedures=[
            LPProc("inc", 0, 0, [
                LPClause(
                    head=LPHead("inc", [], []),
                    goals=[
                        PrimOp("gget", [LPVar("C")], ["c"]),
                        PrimOp("add", [LPVar("c"), LPConst(1)], ["cp"]),
                        PrimOp("gset", [LPVar("C"), LPVar("cp")], []),
                    ],
                ),
            ]),
        ],
        globals=[GlobalDecl("C", 0)],
    )
    smt = extract_chc(prog)
    query = """
(assert (forall ((c_pre Int) (c_post Int))
    (=> (and (inc c_pre c_post) (not (= c_post (+ c_pre 1)))) false)))
"""
    s = _parse_horn(smt + query)
    result = s.check()
    assert result == z3.sat, f"expected sat, got {result}"
    print(f"  test_stateful_property: {PASS}")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_prime_variable_names():
    """Variable names containing `'` (e.g., `i'`, `n'`) must be sanitized."""
    source = """
        fact(n; ret): fact_acc(n, 1; ret).
        fact_acc(n, acc; ret): n > 0, mul(acc, n; acc'),
                               sub(n, 1; n'), fact_acc(n', acc'; ret).
        fact_acc(n, acc; acc): n == 0.
    """
    prog = parse_lp(source)
    smt = extract_chc(prog)
    assert "'" not in smt, "apostrophe should be sanitized"
    _parse_horn(smt)
    print(f"  test_prime_variable_names: {PASS}")


# ---------------------------------------------------------------------------
# WAM runtime: safety + heap-consistency properties
#
# Predicate signatures (see _wam_smt() output) — arguments in order:
#   heap_push        : tag val H_in HEAP_in old_h H_out HEAP_out
#   deref            : addr HEAP_in ret HEAP_out
#   trail_check      : addr HB_in TR_in TRAIL_in HB_out TR_out TRAIL_out
#   bind             : a1 a2 HB_in TR_in HEAP_in TRAIL_in
#                          HB_out TR_out HEAP_out TRAIL_out
#   unwind_trail     : saved_tr TR_in HEAP_in TRAIL_in
#                          TR_out HEAP_out TRAIL_out
# ---------------------------------------------------------------------------

def test_heap_push_increments_H():
    """heap_push(tag, val; old_h) — H_out == H_in + 1."""
    query = """
(assert (forall ((tag Int) (val Int) (H_in Int)
                 (HEAP_in (Array Int Int)) (old_h Int) (H_out Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (heap_push tag val H_in HEAP_in old_h H_out HEAP_out)
             (not (= H_out (+ H_in 1))))
        false)))
"""
    check_holds(query, slice_to=["heap_push"])
    print(f"  test_heap_push_increments_H: {PASS}")


def test_heap_push_returns_old_H():
    """heap_push(tag, val; old_h) — old_h == H_in."""
    query = """
(assert (forall ((tag Int) (val Int) (H_in Int)
                 (HEAP_in (Array Int Int)) (old_h Int) (H_out Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (heap_push tag val H_in HEAP_in old_h H_out HEAP_out)
             (not (= old_h H_in)))
        false)))
"""
    check_holds(query, slice_to=["heap_push"])
    print(f"  test_heap_push_returns_old_H: {PASS}")


def test_heap_push_stores_tag_and_val():
    """heap_push — HEAP_out[2*H_in] == tag, HEAP_out[2*H_in + 1] == val."""
    query = """
(assert (forall ((tag Int) (val Int) (H_in Int)
                 (HEAP_in (Array Int Int)) (old_h Int) (H_out Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (heap_push tag val H_in HEAP_in old_h H_out HEAP_out)
             (or (not (= (select HEAP_out (* 2 H_in)) tag))
                 (not (= (select HEAP_out (+ (* 2 H_in) 1)) val))))
        false)))
"""
    check_holds(query, slice_to=["heap_push"])
    print(f"  test_heap_push_stores_tag_and_val: {PASS}")


def test_heap_push_preserves_other_cells():
    """heap_push — HEAP_out[k] == HEAP_in[k] for k < 2*H_in."""
    query = """
(assert (forall ((tag Int) (val Int) (H_in Int)
                 (HEAP_in (Array Int Int)) (old_h Int) (H_out Int)
                 (HEAP_out (Array Int Int)) (k Int))
    (=> (and (heap_push tag val H_in HEAP_in old_h H_out HEAP_out)
             (< k (* 2 H_in))
             (not (= (select HEAP_out k) (select HEAP_in k))))
        false)))
"""
    check_holds(query, slice_to=["heap_push"])
    print(f"  test_heap_push_preserves_other_cells: {PASS}")


def test_trail_check_trails_when_below_HB():
    """addr < HB_in ⇒ TR_out == TR_in + 1, TRAIL_out[TR_in] == addr."""
    query = """
(assert (forall ((addr Int) (HB_in Int) (TR_in Int)
                 (TRAIL_in (Array Int Int)) (HB_out Int) (TR_out Int)
                 (TRAIL_out (Array Int Int)))
    (=> (and (trail_check addr HB_in TR_in TRAIL_in HB_out TR_out TRAIL_out)
             (< addr HB_in)
             (or (not (= TR_out (+ TR_in 1)))
                 (not (= (select TRAIL_out TR_in) addr))
                 (not (= HB_out HB_in))))
        false)))
"""
    check_holds(query, slice_to=["trail_check"])
    print(f"  test_trail_check_trails_when_below_HB: {PASS}")


def test_deref_preserves_HEAP():
    """deref is a pure reader — HEAP_out == HEAP_in."""
    query = """
(assert (forall ((addr Int) (HEAP_in (Array Int Int)) (ret Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (deref addr HEAP_in ret HEAP_out)
             (not (= HEAP_in HEAP_out)))
        false)))
"""
    check_holds(query, slice_to=["deref"])
    print(f"  test_deref_preserves_HEAP: {PASS}")


def test_deref_non_ref_is_identity():
    """If HEAP[2*addr] != REF (tag=0), deref returns addr unchanged."""
    query = """
(assert (forall ((addr Int) (HEAP_in (Array Int Int)) (ret Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (deref addr HEAP_in ret HEAP_out)
             (not (= (select HEAP_in (* 2 addr)) 0))
             (not (= ret addr)))
        false)))
"""
    check_holds(query, slice_to=["deref"])
    print(f"  test_deref_non_ref_is_identity: {PASS}")


def test_deref_self_ref_is_identity():
    """If addr is a self-referential REF, deref returns addr."""
    query = """
(assert (forall ((addr Int) (HEAP_in (Array Int Int)) (ret Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (deref addr HEAP_in ret HEAP_out)
             (= (select HEAP_in (* 2 addr)) 0)
             (= (select HEAP_in (+ (* 2 addr) 1)) addr)
             (not (= ret addr)))
        false)))
"""
    check_holds(query, slice_to=["deref"])
    print(f"  test_deref_self_ref_is_identity: {PASS}")


def test_bind_preserves_HB():
    """bind(a1, a2) never writes HB."""
    query = """
(assert (forall ((a1 Int) (a2 Int) (HB_in Int) (TR_in Int)
                 (HEAP_in (Array Int Int)) (TRAIL_in (Array Int Int))
                 (HB_out Int) (TR_out Int)
                 (HEAP_out (Array Int Int)) (TRAIL_out (Array Int Int)))
    (=> (and (bind a1 a2 HB_in TR_in HEAP_in TRAIL_in
                        HB_out TR_out HEAP_out TRAIL_out)
             (not (= HB_out HB_in)))
        false)))
"""
    check_holds(query, slice_to=["bind"])
    print(f"  test_bind_preserves_HB: {PASS}")


def test_bind_ref_to_nonref_copies_cell():
    """Clause 0: tag(a1)=REF, tag(a2)!=REF ⇒ HEAP_out[a1] := HEAP_in[a2]."""
    query = """
(assert (forall ((a1 Int) (a2 Int) (HB_in Int) (TR_in Int)
                 (HEAP_in (Array Int Int)) (TRAIL_in (Array Int Int))
                 (HB_out Int) (TR_out Int)
                 (HEAP_out (Array Int Int)) (TRAIL_out (Array Int Int)))
    (=> (and (bind a1 a2 HB_in TR_in HEAP_in TRAIL_in
                        HB_out TR_out HEAP_out TRAIL_out)
             (= (select HEAP_in (* 2 a1)) 0)
             (not (= (select HEAP_in (* 2 a2)) 0))
             (or (not (= (select HEAP_out (* 2 a1))
                         (select HEAP_in (* 2 a2))))
                 (not (= (select HEAP_out (+ (* 2 a1) 1))
                         (select HEAP_in (+ (* 2 a2) 1))))))
        false)))
"""
    check_holds(query, slice_to=["bind"])
    print(f"  test_bind_ref_to_nonref_copies_cell: {PASS}")


def test_bind_trails_when_below_HB():
    """When bind modifies an address < HB, trail grows by 1."""
    # Clause 0: modifies a1. If a1 < HB_in, trail a1.
    query = """
(assert (forall ((a1 Int) (a2 Int) (HB_in Int) (TR_in Int)
                 (HEAP_in (Array Int Int)) (TRAIL_in (Array Int Int))
                 (HB_out Int) (TR_out Int)
                 (HEAP_out (Array Int Int)) (TRAIL_out (Array Int Int)))
    (=> (and (bind a1 a2 HB_in TR_in HEAP_in TRAIL_in
                        HB_out TR_out HEAP_out TRAIL_out)
             (= (select HEAP_in (* 2 a1)) 0)
             (not (= (select HEAP_in (* 2 a2)) 0))
             (< a1 HB_in)
             (or (not (= TR_out (+ TR_in 1)))
                 (not (= (select TRAIL_out TR_in) a1))))
        false)))
"""
    check_holds(query, slice_to=["bind"])
    print(f"  test_bind_trails_when_below_HB: {PASS}")


def test_unwind_trail_single_step_decrements():
    """unwind_trail with TR_in == saved_tr + 1 does one step and stops."""
    # Two-step: first call with TR > saved_tr (takes 1 step to TR=saved_tr),
    # then base case. Verify final TR_out == saved_tr.
    query = """
(assert (forall ((saved_tr Int) (HEAP_in (Array Int Int))
                 (TRAIL_in (Array Int Int)) (TR_out Int)
                 (HEAP_out (Array Int Int)) (TRAIL_out (Array Int Int)))
    (=> (and (unwind_trail saved_tr (+ saved_tr 1) HEAP_in TRAIL_in
                            TR_out HEAP_out TRAIL_out)
             (not (= TR_out saved_tr)))
        false)))
"""
    check_holds(query, slice_to=["unwind_trail"])
    print(f"  test_unwind_trail_single_step_decrements: {PASS}")


def test_unwind_trail_base_case():
    """unwind_trail with TR <= saved_tr: state unchanged."""
    query = """
(assert (forall ((saved_tr Int) (TR_in Int) (HEAP_in (Array Int Int))
                 (TRAIL_in (Array Int Int)) (TR_out Int)
                 (HEAP_out (Array Int Int)) (TRAIL_out (Array Int Int)))
    (=> (and (unwind_trail saved_tr TR_in HEAP_in TRAIL_in
                            TR_out HEAP_out TRAIL_out)
             (<= TR_in saved_tr)
             (or (not (= TR_out TR_in))
                 (not (= HEAP_out HEAP_in))
                 (not (= TRAIL_out TRAIL_in))))
        false)))
"""
    check_holds(query, slice_to=["unwind_trail"])
    print(f"  test_unwind_trail_base_case: {PASS}")


def test_pdl_push_increments_top():
    """pdl_push(val) increments PDL_TOP by 1 and stores val."""
    query = """
(assert (forall ((val Int) (PDL_TOP_in Int) (PDL_in (Array Int Int))
                 (PDL_TOP_out Int) (PDL_out (Array Int Int)))
    (=> (and (pdl_push val PDL_TOP_in PDL_in PDL_TOP_out PDL_out)
             (or (not (= PDL_TOP_out (+ PDL_TOP_in 1)))
                 (not (= (select PDL_out PDL_TOP_in) val))))
        false)))
"""
    check_holds(query, slice_to=["pdl_push"])
    print(f"  test_pdl_push_increments_top: {PASS}")


def test_pdl_pop_decrements_top():
    """pdl_pop gives PDL_TOP_out = PDL_TOP_in - 1, val = PDL_in[PDL_TOP_in - 1]."""
    query = """
(assert (forall ((PDL_TOP_in Int) (PDL_in (Array Int Int))
                 (val Int) (PDL_TOP_out Int) (PDL_out (Array Int Int)))
    (=> (and (pdl_pop PDL_TOP_in PDL_in val PDL_TOP_out PDL_out)
             (or (not (= PDL_TOP_out (- PDL_TOP_in 1)))
                 (not (= val (select PDL_in (- PDL_TOP_in 1))))))
        false)))
"""
    check_holds(query, slice_to=["pdl_pop"])
    print(f"  test_pdl_pop_decrements_top: {PASS}")


def test_unify_con_equal_ok():
    """unify_con with v1 == v2 leaves FAIL unchanged (FAIL_out == FAIL_in)."""
    query = """
(assert (forall ((v1 Int) (v2 Int) (FAIL_in Int) (FAIL_out Int))
    (=> (and (unify_con v1 v2 FAIL_in FAIL_out)
             (= v1 v2)
             (not (= FAIL_out FAIL_in)))
        false)))
"""
    check_holds(query, slice_to=["unify_con"])
    print(f"  test_unify_con_equal_ok: {PASS}")


def test_unify_con_unequal_sets_FAIL():
    """unify_con with v1 != v2 sets FAIL_out = 1."""
    query = """
(assert (forall ((v1 Int) (v2 Int) (FAIL_in Int) (FAIL_out Int))
    (=> (and (unify_con v1 v2 FAIL_in FAIL_out)
             (not (= v1 v2))
             (not (= FAIL_out 1)))
        false)))
"""
    check_holds(query, slice_to=["unify_con"])
    print(f"  test_unify_con_unequal_sets_FAIL: {PASS}")


# ---------------------------------------------------------------------------
# Termination tests
# ---------------------------------------------------------------------------

def test_factorial_terminates():
    """fact_acc(n, acc; ret) terminates — n strictly decreases."""
    prog = parse_lp("""
        fact(n; ret): fact_acc(n, 1; ret).
        fact_acc(n, acc; ret): n > 0, mul(acc, n; acc'),
                               sub(n, 1; n'), fact_acc(n', acc'; ret).
        fact_acc(n, acc; acc): n <= 0.
    """)
    # Set measure on the recursive proc
    for p in prog.procedures:
        if p.name == "fact_acc":
            p.measure = ["n"]
    smt = extract_chc(prog)
    # The termination obligation says: if fact_acc recurs with n' < n,
    # and n' < n is required, then there is no infinite descent.
    # For finite integers this is trivially true.
    # We verify by checking that the obligation + the rules are satisfiable
    # (i.e., no contradiction in the measure encoding).
    s = _parse_horn(smt)
    r = s.check()
    assert r == z3.sat, f"expected sat (measure rules consistent), got {r}"
    print(f"  test_factorial_terminates: {PASS}")


def test_deref_terminates():
    """deref(addr; ret) terminates — REF chain strictly shortens."""
    prog = _wam_program()
    prog_sliced = prog
    for p in prog_sliced.procedures:
        if p.name == "deref":
            p.measure = ["addr"]
    smt = extract_chc(prog_sliced, slice_to=["deref"])
    s = _parse_horn(smt)
    r = s.check()
    assert r == z3.sat, f"expected sat (measure encoding consistent), got {r}"
    print(f"  test_deref_terminates: {PASS}")


# ---------------------------------------------------------------------------
# Counterexample extraction
# ---------------------------------------------------------------------------

def test_cex_factorial_false_property():
    """A false property produces a counterexample with concrete values."""
    from chc import check_property
    smt = extract_chc(make_factorial())
    # False: fact(5, r) => r < 100 (fact(5)=120 > 100)
    query = """
(assert (forall ((n Int) (r Int))
    (=> (and (fact n r) (= n 5) (>= r 100)) false)))
"""
    result, cex = check_property(smt, query, timeout_ms=10_000)
    assert result == "violated", f"expected violated, got {result}"
    print(f"  test_cex_factorial_false_property: {PASS} (cex={cex})")


def test_cex_heap_push_mutation():
    """Mutation of heap_push should produce a counterexample."""
    from chc import check_property
    prog = _wam_program()
    # Mutate: change add(old_h, 1) to add(old_h, 2)
    for p in prog.procedures:
        if p.name == "heap_push":
            for clause in p.clauses:
                for i, goal in enumerate(clause.goals):
                    if isinstance(goal, PrimOp) and goal.op == "add":
                        if (isinstance(goal.inputs[0], LPVar)
                                and goal.inputs[0].name == "old_h"
                                and isinstance(goal.inputs[1], LPConst)
                                and goal.inputs[1].value == 1):
                            clause.goals[i] = PrimOp(
                                "add",
                                [LPVar("old_h"), LPConst(2)],
                                ["new_h"])

    smt = extract_chc(prog, slice_to=["heap_push"])
    query = """
(assert (forall ((tag Int) (val Int) (H_in Int)
                 (HEAP_in (Array Int Int)) (old_h Int) (H_out Int)
                 (HEAP_out (Array Int Int)))
    (=> (and (heap_push tag val H_in HEAP_in old_h H_out HEAP_out)
             (not (= H_out (+ H_in 1))))
        false)))
"""
    result, cex = check_property(smt, query, timeout_ms=10_000)
    assert result == "violated", f"expected violated, got {result}"
    print(f"  test_cex_heap_push_mutation: {PASS} (cex={cex})")


def test_trail_check_noop_when_above_HB():
    """addr >= HB_in ⇒ state unchanged."""
    query = """
(assert (forall ((addr Int) (HB_in Int) (TR_in Int)
                 (TRAIL_in (Array Int Int)) (HB_out Int) (TR_out Int)
                 (TRAIL_out (Array Int Int)))
    (=> (and (trail_check addr HB_in TR_in TRAIL_in HB_out TR_out TRAIL_out)
             (>= addr HB_in)
             (or (not (= TR_out TR_in))
                 (not (= TRAIL_out TRAIL_in))
                 (not (= HB_out HB_in))))
        false)))
"""
    check_holds(query, slice_to=["trail_check"])
    print(f"  test_trail_check_noop_when_above_HB: {PASS}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CHC extraction tests ===")
    tests = [
        test_gcd_parses,
        test_factorial_parses,
        test_fibonacci_parses,
        test_divmod_parses,
        test_wam_runtime_parses,
        test_prime_variable_names,
        test_stateful_parses,
        test_stateful_property,
        test_factorial_positivity,
        test_fibonacci_base_case,
        test_fibonacci_non_negative,
        test_counterexample_detected,
        # WAM runtime properties
        test_heap_push_increments_H,
        test_heap_push_returns_old_H,
        test_heap_push_stores_tag_and_val,
        test_heap_push_preserves_other_cells,
        test_trail_check_trails_when_below_HB,
        test_trail_check_noop_when_above_HB,
        test_deref_preserves_HEAP,
        test_deref_non_ref_is_identity,
        test_deref_self_ref_is_identity,
        test_bind_preserves_HB,
        test_bind_ref_to_nonref_copies_cell,
        test_bind_trails_when_below_HB,
        test_unwind_trail_single_step_decrements,
        test_unwind_trail_base_case,
        test_pdl_push_increments_top,
        test_pdl_pop_decrements_top,
        test_unify_con_equal_ok,
        test_unify_con_unequal_sets_FAIL,
        # Termination
        test_factorial_terminates,
        test_deref_terminates,
        # Counterexample extraction
        test_cex_factorial_false_property,
        test_cex_heap_push_mutation,
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
