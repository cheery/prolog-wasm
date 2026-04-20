"""Tests for CHC extraction from LP Form programs.

Extracts CHC from LP Form programs, parses with Z3, and verifies
simple safety properties where tractable.

Note on Z3/Spacer results:
  - sat   = property holds (no counterexample)
  - unsat = property violated (counterexample exists)
  - unknown = solver couldn't decide (common with mod, arrays)
"""

import sys, os
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

    # Sanity: heap_push should take H_in, HEAP_in, H_out, HEAP_out
    # (plus the explicit tag, val, old_h args)
    assert "(declare-fun heap_push " in smt
    # init should thread all globals and i32 arrays
    assert "(declare-fun init " in smt
    # ref arrays (CONT, BP_STACK) must NOT appear as predicate args
    assert "CONT" not in smt.split("(assert")[0]  # only in declarations section
    assert "BP_STACK" not in smt.split("(assert")[0]
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
