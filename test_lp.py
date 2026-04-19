"""Tests for the LP Form -> WASM pipeline.

Compiles LP Form programs to WASM, runs them with wasmtime, checks results.
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from lp_form import *
from lp_pipeline import lp_compile

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

WASMTIME = "/home/cheery/.wasmtime/bin/wasmtime"


def validate_wasm(wasm_bytes, label):
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


def run_wasm(wasm_bytes, func_name, *args):
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
        # Parse output — may be multi-value
        line = r.stdout.strip()
        parts = line.split()
        if len(parts) == 1:
            return int(parts[0])
        return tuple(int(p) for p in parts)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 1: GCD — the paper's canonical example (Figure 8)
#
#   gcd(a, b; ret) <- b != 0 /\ mod(a, b; b') /\ gcd(b, b'; ret)
#   gcd(a, b; ret) <- b = 0  /\ copy(a; ret)
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


def test_gcd():
    prog = make_gcd()
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "gcd"), "validation failed"

    cases = [
        ((12, 8), 4),
        ((100, 75), 25),
        ((7, 0), 7),
        ((0, 5), 5),
        ((17, 13), 1),
        ((48, 36), 12),
    ]
    for (a, b), expected in cases:
        result = run_wasm(wasm, "run", a, b)
        assert result == expected, f"gcd({a},{b}): expected {expected}, got {result}"
    print(f"  test_gcd: {PASS}")


# ---------------------------------------------------------------------------
# Test 2: Factorial with accumulator
#
#   fact(n; ret) <- fact_acc(n, 1; ret)
#   fact_acc(n, acc; ret) <- n > 0 /\ mul(acc, n; acc') /\ sub(n, 1; n')
#                            /\ fact_acc(n', acc'; ret)
#   fact_acc(n, acc; ret) <- n = 0 /\ copy(acc; ret)
# ---------------------------------------------------------------------------

def make_factorial():
    return LPProgram(
        procedures=[
            LPProc("fact", 1, 1, [
                LPClause(
                    head=LPHead("fact", ["n"], ["ret"]),
                    goals=[
                        Call("fact_acc", [LPVar("n"), LPConst(1)], ["ret"]),
                    ],
                ),
            ]),
            LPProc("fact_acc", 2, 1, [
                LPClause(
                    head=LPHead("fact_acc", ["n", "acc"], ["ret"]),
                    goals=[
                        Guard("gt", LPVar("n"), LPConst(0)),
                        PrimOp("mul", [LPVar("acc"), LPVar("n")], ["acc_prime"]),
                        PrimOp("sub", [LPVar("n"), LPConst(1)], ["n_prime"]),
                        Call("fact_acc",
                             [LPVar("n_prime"), LPVar("acc_prime")], ["ret"]),
                    ],
                ),
                LPClause(
                    head=LPHead("fact_acc", ["n", "acc"], ["ret"]),
                    goals=[
                        Guard("le", LPVar("n"), LPConst(0)),
                        PrimOp("copy", [LPVar("acc")], ["ret"]),
                    ],
                ),
            ]),
        ],
        entry="fact",
    )


def test_factorial():
    prog = make_factorial()
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "factorial"), "validation failed"

    cases = [(0, 1), (1, 1), (5, 120), (10, 3628800)]
    for n, expected in cases:
        result = run_wasm(wasm, "run", n)
        assert result == expected, f"fact({n}): expected {expected}, got {result}"
    print(f"  test_factorial: {PASS}")


# ---------------------------------------------------------------------------
# Test 3: Fibonacci (non-tail recursive)
#
#   fib(n; ret) <- n <= 1 /\ copy(n; ret)
#   fib(n; ret) <- n > 1  /\ sub(n,1;a) /\ sub(n,2;b)
#                  /\ fib(a;x) /\ fib(b;y) /\ add(x,y;ret)
# ---------------------------------------------------------------------------

def make_fibonacci():
    return LPProgram(
        procedures=[
            LPProc("fib", 1, 1, [
                LPClause(
                    head=LPHead("fib", ["n"], ["ret"]),
                    goals=[
                        Guard("le", LPVar("n"), LPConst(1)),
                        PrimOp("copy", [LPVar("n")], ["ret"]),
                    ],
                ),
                LPClause(
                    head=LPHead("fib", ["n"], ["ret"]),
                    goals=[
                        Guard("gt", LPVar("n"), LPConst(1)),
                        PrimOp("sub", [LPVar("n"), LPConst(1)], ["a"]),
                        PrimOp("sub", [LPVar("n"), LPConst(2)], ["b"]),
                        Call("fib", [LPVar("a")], ["x"]),
                        Call("fib", [LPVar("b")], ["y"]),
                        PrimOp("add", [LPVar("x"), LPVar("y")], ["ret"]),
                    ],
                ),
            ]),
        ],
        entry="fib",
    )


def test_fibonacci():
    prog = make_fibonacci()
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "fibonacci"), "validation failed"

    cases = [(0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (15, 610)]
    for n, expected in cases:
        result = run_wasm(wasm, "run", n)
        assert result == expected, f"fib({n}): expected {expected}, got {result}"
    print(f"  test_fibonacci: {PASS}")


# ---------------------------------------------------------------------------
# Test 4: Divmod — multi-value return
#
#   divmod(a, b; q, r) <- div(a, b; q) /\ rem(a, b; r)
# ---------------------------------------------------------------------------

def make_divmod():
    return LPProgram(
        procedures=[
            LPProc("divmod", 2, 2, [
                LPClause(
                    head=LPHead("divmod", ["a", "b"], ["q", "r"]),
                    goals=[
                        PrimOp("div", [LPVar("a"), LPVar("b")], ["q"]),
                        PrimOp("rem", [LPVar("a"), LPVar("b")], ["r"]),
                    ],
                ),
            ]),
        ],
        entry=None,  # export all
    )


def test_divmod():
    prog = make_divmod()
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "divmod"), "validation failed"

    cases = [
        ((17, 5), (3, 2)),
        ((100, 7), (14, 2)),
        ((10, 2), (5, 0)),
    ]
    for (a, b), (eq, er) in cases:
        result = run_wasm(wasm, "divmod", a, b)
        assert result == (eq, er), \
            f"divmod({a},{b}): expected ({eq},{er}), got {result}"
    print(f"  test_divmod: {PASS}")


# ---------------------------------------------------------------------------
# Test 5: Pretty printer
# ---------------------------------------------------------------------------

def test_pretty_print():
    prog = make_gcd()
    text = pretty_print(prog)
    assert "gcd" in text
    assert "!=" in text or "ne" in text
    assert "rem" in text
    print(f"  test_pretty_print: {PASS}")


# ---------------------------------------------------------------------------
# Test 5: Validation catches errors
# ---------------------------------------------------------------------------

def test_validation_errors():
    # Duplicate assignment
    bad = LPProgram(procedures=[
        LPProc("bad", 1, 1, [
            LPClause(
                head=LPHead("bad", ["x"], ["ret"]),
                goals=[
                    PrimOp("copy", [LPVar("x")], ["ret"]),
                    PrimOp("copy", [LPVar("x")], ["ret"]),  # double assign
                ],
            ),
        ]),
    ])
    try:
        validate(bad)
        assert False, "should have raised"
    except ValueError:
        pass

    # Undefined output
    bad2 = LPProgram(procedures=[
        LPProc("bad2", 1, 1, [
            LPClause(
                head=LPHead("bad2", ["x"], ["ret"]),
                goals=[],  # ret never defined
            ),
        ]),
    ])
    try:
        validate(bad2)
        assert False, "should have raised"
    except ValueError:
        pass

    print(f"  test_validation_errors: {PASS}")


# ---------------------------------------------------------------------------
# Test 7: Parse + compile from LP Form source text
# ---------------------------------------------------------------------------

def test_parse_gcd():
    from lp_parser import parse_lp

    source = """
        gcd(a, b; ret): b != 0, mod(a, b; b'), gcd(b, b'; ret).
        gcd(a, b; a): b == 0.
    """
    prog = parse_lp(source)
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "parse_gcd"), "validation failed"

    cases = [((12, 8), 4), ((100, 75), 25), ((17, 13), 1)]
    for (a, b), expected in cases:
        result = run_wasm(wasm, "run", a, b)
        assert result == expected, f"gcd({a},{b}): expected {expected}, got {result}"
    print(f"  test_parse_gcd: {PASS}")


def test_parse_factorial():
    from lp_parser import parse_lp

    source = """
        fact(n; ret): fact_acc(n, 1; ret).
        fact_acc(n, acc; ret): n > 0, mul(acc, n; acc'),
                               sub(n, 1; n'), fact_acc(n', acc'; ret).
        fact_acc(n, acc; acc): n == 0.
    """
    prog = parse_lp(source)
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "parse_factorial"), "validation failed"

    cases = [(0, 1), (1, 1), (5, 120), (10, 3628800)]
    for n, expected in cases:
        result = run_wasm(wasm, "run", n)
        assert result == expected, f"fact({n}): expected {expected}, got {result}"
    print(f"  test_parse_factorial: {PASS}")


def test_parse_fibonacci():
    from lp_parser import parse_lp

    source = """
        fib(n; ret): n <= 1, copy(n; ret).
        fib(n; ret): n > 1, sub(n, 1; a), sub(n, 2; b),
                     fib(a; x), fib(b; y), add(x, y; ret).
    """
    prog = parse_lp(source)
    wasm = lp_compile(prog)
    assert validate_wasm(wasm, "parse_fibonacci"), "validation failed"

    cases = [(0, 0), (1, 1), (5, 5), (10, 55)]
    for n, expected in cases:
        result = run_wasm(wasm, "run", n)
        assert result == expected, f"fib({n}): expected {expected}, got {result}"
    print(f"  test_parse_fibonacci: {PASS}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LP Form tests ===")
    tests = [
        test_pretty_print,
        test_validation_errors,
        test_gcd,
        test_factorial,
        test_fibonacci,
        test_divmod,
        test_parse_gcd,
        test_parse_factorial,
        test_parse_fibonacci,
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
