"""Phase 6 tests: full Prolog → WASM pipeline via plwasm.compile().

Each test compiles a Prolog program with plwasm.compile(), runs it with
run_wasm(bytes, "run_get", reg_index), and checks the result.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from plwasm import compile as pl_compile
from wam_wasm import validate_wasm, run_wasm

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def run_get(wasm, n):
    return run_wasm(wasm, "run_get", n)


# ---------------------------------------------------------------------------
# Test 1: Simple numeric fact
# ---------------------------------------------------------------------------

def test_answer_42():
    source = """
        answer(42).
        ?- answer(X).
    """
    wasm, _syms = pl_compile(source)
    assert validate_wasm(wasm, "answer(42)"), "validation failed"
    result = run_get(wasm, 1)
    assert result == 42, f"expected 42, got {result}"
    print(f"  test_answer_42: {PASS}")


# ---------------------------------------------------------------------------
# Test 2: Atom constant fact
# ---------------------------------------------------------------------------

def test_greeting_atom():
    source = """
        greeting(hello).
        ?- greeting(X).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "greeting(hello)"), "validation failed"
    result = run_get(wasm, 1)
    expected = syms.encode_constant("hello")
    assert result == expected, f"expected hello={expected}, got {result}"
    print(f"  test_greeting_atom: {PASS}")


# ---------------------------------------------------------------------------
# Test 3: Three-clause backtracking — find matching atom
# ---------------------------------------------------------------------------

def test_color_backtrack():
    source = """
        color(red).
        color(green).
        color(blue).
        ?- color(blue).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "color(blue)"), "validation failed"
    result = run_get(wasm, 1)
    expected = syms.encode_constant("blue")
    assert result == expected, f"expected blue={expected}, got {result}"
    print(f"  test_color_backtrack: {PASS}")


# ---------------------------------------------------------------------------
# Test 4: Backtracking past a failing clause to reach a succeeding one
# ---------------------------------------------------------------------------

def test_backtrack_past_failure():
    source = """
        result(bad).
        result(42).
        ?- result(42).
    """
    wasm, _syms = pl_compile(source)
    assert validate_wasm(wasm, "result(42)"), "validation failed"
    result = run_get(wasm, 1)
    assert result == 42, f"expected 42, got {result}"
    print(f"  test_backtrack_past_failure: {PASS}")


# ---------------------------------------------------------------------------
# Test 5: neck_cut — only the first alternative is returned
# ---------------------------------------------------------------------------

def test_neck_cut():
    source = """
        first(1).
        first(2).
        first(3).
        only_first(X) :- first(X), !.
        ?- only_first(X).
    """
    wasm, _syms = pl_compile(source)
    assert validate_wasm(wasm, "neck_cut"), "validation failed"
    result = run_get(wasm, 1)
    assert result == 1, f"expected 1, got {result}"
    print(f"  test_neck_cut: {PASS}")


# ---------------------------------------------------------------------------
# Test 6: Rule with two-goal body
# ---------------------------------------------------------------------------

def test_two_goal_rule():
    source = """
        age(alice, 30).
        name(alice, alice_name).
        person(Name, Age) :- age(alice, Age), name(alice, Name).
        ?- person(N, A).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "two_goal_rule"), "validation failed"
    # Compiler assigns: ?- person(N, A) → N=X4, A=X3 (see compiled output)
    # After execution X3 holds the Age binding (30)
    age_result = run_get(wasm, 3)
    assert age_result == 30, f"expected Age=30, got {age_result}"
    print(f"  test_two_goal_rule: {PASS}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 6 tests ===")
    tests = [
        test_answer_42,
        test_greeting_atom,
        test_color_backtrack,
        test_backtrack_past_failure,
        test_neck_cut,
        test_two_goal_rule,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: {FAIL} — {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
