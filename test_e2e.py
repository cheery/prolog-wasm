"""End-to-end test: Prolog -> LP Form -> WASM -> run.

Compiles Prolog programs through the unified pipeline (plwasm.compile)
and validates results with wasmtime.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from plwasm import compile as pl_compile
from wam_wasm import validate_wasm, run_wasm


def run_get(wasm, n):
    return run_wasm(wasm, "run_get", n)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def test_single_fact():
    """Compilation succeeds for a fact with no query."""
    source = """
        parent(tom, bob).
    """
    wasm, _ = pl_compile(source)
    assert validate_wasm(wasm, "single fact")
    print(f"  test_single_fact: {PASS}")


def test_answer_42():
    source = """
        answer(42).
        ?- answer(X).
    """
    wasm, _ = pl_compile(source)
    assert validate_wasm(wasm, "answer(42)")
    assert run_get(wasm, 1) == 42
    print(f"  test_answer_42: {PASS}")


def test_greeting_atom():
    source = """
        greeting(hello).
        ?- greeting(X).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "greeting(hello)")
    result = run_get(wasm, 1)
    expected = syms.encode_constant("hello")
    assert result == expected, f"expected {expected}, got {result}"
    print(f"  test_greeting_atom: {PASS}")


def test_backtracking():
    source = """
        result(bad).
        result(42).
        ?- result(42).
    """
    wasm, _ = pl_compile(source)
    assert validate_wasm(wasm, "backtracking")
    assert run_get(wasm, 1) == 42
    print(f"  test_backtracking: {PASS}")


def test_three_clause_backtrack():
    source = """
        color(red).
        color(green).
        color(blue).
        ?- color(blue).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "three-clause backtrack")
    result = run_get(wasm, 1)
    blue_id = syms.encode_constant("blue")
    assert result == blue_id, f"expected {blue_id}, got {result}"
    print(f"  test_three_clause_backtrack: {PASS}")


def test_neck_cut():
    source = """
        first(1).
        first(2).
        first(3).
        only_first(X) :- first(X), !.
        ?- only_first(X).
    """
    wasm, _ = pl_compile(source)
    assert validate_wasm(wasm, "neck_cut")
    assert run_get(wasm, 1) == 1
    print(f"  test_neck_cut: {PASS}")


def test_two_goal_rule():
    source = """
        age(alice, 30).
        name(alice, alice_name).
        person(Name, Age) :- age(alice, Age), name(alice, Name).
        ?- person(N, A).
    """
    wasm, syms = pl_compile(source)
    assert validate_wasm(wasm, "two_goal_rule")
    age_result = run_get(wasm, 3)
    assert age_result == 30, f"expected Age=30, got {age_result}"
    print(f"  test_two_goal_rule: {PASS}")


TESTS = [
    test_single_fact,
    test_answer_42,
    test_greeting_atom,
    test_backtracking,
    test_three_clause_backtrack,
    test_neck_cut,
    test_two_goal_rule,
]


if __name__ == "__main__":
    print("=== Prolog end-to-end (via LP Form) ===")
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
    if failed:
        sys.exit(1)
