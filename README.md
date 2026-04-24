# wam

A playground for **LP Form** — a small, deterministic logic-programming IR
that compiles to WebAssembly (with GC) and extracts directly to
Constrained Horn Clauses for verification with Z3/Spacer.

The repository started as a Prolog-to-WASM compiler and the Prolog
front-end is still here (see *Prolog, as a secondary frontend* below),
but the interesting work has moved to LP Form: a language where the
**same source** compiles to efficient WASM *and* extracts to CHC for
formal verification, without two separate semantic models.

## What is LP Form?

LP Form is a restricted, single-moded logic-programming language based
on Gange et al. (TPLP 2015). The shape is familiar:

```
gcd(a, b; ret): b != 0, rem(a, b; b'), gcd(b, b'; ret).
gcd(a, b; a):   b == 0.
```

Key restrictions that make compilation *and* verification easy:

- **Single-moded**: each argument is either an input (value) or an output
  (fresh variable). No free/bound inference, no backtracking.
- **Complementary guards**: exactly one clause fires per call. Multi-clause
  dispatch is manual pattern-matching on guards, so control flow is
  deterministic.
- **Loops are tail recursion**. Tail calls are marked automatically and
  emitted as WASM `return_call`.
- **State is explicit**: mutable i32 globals and GC arrays are declared
  up front and manipulated through named PrimOps (`gget`/`gset`,
  `aget`/`aset`/`anew`).

Because the language is already a set of Horn clauses, extracting CHC
for a verifier is structural, not semantic — each procedure becomes an
uninterpreted predicate, each clause becomes a `(forall … body ⇒ head)`
rule.

## Repository layout

| File | Description |
|---|---|
| `lp_form.py` | IR dataclasses, validator, pretty-printer, output type inference |
| `lp_parser.py` | Lark parser for LP Form source syntax |
| `lp_elaborate.py` | Type elaboration pass (patterns → guards, field access → struct_get) |
| `lp_emit.py` | LP Form → WASM emitter (with ref-typed returns) |
| `lp_pipeline.py` | `validate → infer_output_types → elaborate → mark_tail_calls → emit` |
| `chc.py` | LP Form → SMT-LIB2 CHC extractor |
| `wam_runtime.lp` | WAM runtime written in LP Form |
| `wasm/encoder.py` | Hand-rolled WASM binary encoder |
| `plwasm.py` | Prolog → LP Form → WASM unified pipeline |
| `prolog_to_lp.py` | Prolog → LP Form lowering |
| `test_lp.py` | Compile LP programs and run in wasmtime |
| `test_chc.py` | Extract CHC and check properties with Z3 |
| `test_types.py` | Type system: structs, sums, ADTs, cross-proc ref returns |
| `test_wam_runtime.py` | Compile `wam_runtime.lp` end-to-end |
| `test_e2e.py` | Prolog → LP Form → WASM end-to-end |

## Requirements

- Python 3.10+
- `pip install lark z3-solver`
- [`wasmtime`](https://wasmtime.dev/) on `$PATH` (any recent version
  with the GC / function-references proposals enabled — the tests pass
  `-W all-proposals=y`).

## Tutorial

### 1. Compile an LP Form program to WASM

Start with Euclid's gcd. The program is two clauses, one for each
branch of the guard:

```python
from lp_form import *
from lp_pipeline import lp_compile

program = LPProgram(
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
                head=LPHead("gcd", ["a", "b"], ["a"]),
                goals=[Guard("eq", LPVar("b"), LPConst(0))],
            ),
        ]),
    ],
    entry="gcd",
)

wasm_bytes = lp_compile(program)
```

Or, equivalently, write it as source and parse:

```python
from lp_parser import parse_lp

program = parse_lp("""
    gcd(a, b; ret): b != 0, rem(a, b; b'), gcd(b, b'; ret).
    gcd(a, b; a):   b == 0.
""")
wasm_bytes = lp_compile(program)
```

The emitted module exports a `run` function. Invoke it:

```bash
wasmtime -W all-proposals=y --invoke run ./gcd.wasm 100 75
# 25
```

The recursive call in the first clause is in tail position, so it
compiles to `return_call` — gcd of arbitrarily large inputs runs in
constant stack.

### 2. State: globals and arrays

LP Form has first-class mutable state, declared at the top of the file:

```
global  H    = 0.
array   HEAP.
array   CONT: ref.   // funcref array (for continuations)

heap_push(tag, val; addr):
    gget(H; addr),
    mul(addr, 2; idx),
    aset(HEAP, idx, tag;),
    add(idx, 1; idx1),
    aset(HEAP, idx1, val;),
    add(addr, 1; new_h),
    gset(H, new_h;).
```

The emitter maps i32 arrays to WASM GC `(array i32)` types, funcref
arrays to `(array (ref null func))`, and globals to WASM mutable
globals. See `wam_runtime.lp` for a complete example: the entire
Warren Abstract Machine runtime (heap, trail, PDL, unification,
backtracking) is ~320 lines of LP Form.

### 3. Extract CHC and verify with Z3

LP Form's structure *is* Horn clauses, so extraction is mechanical:

```python
from chc import extract_chc
from lp_parser import parse_lp

program = parse_lp("""
    fact(n; ret): fact_acc(n, 1; ret).
    fact_acc(n, acc; ret): n > 0,
        mul(acc, n; acc'), sub(n, 1; n'),
        fact_acc(n', acc'; ret).
    fact_acc(n, acc; acc): n <= 0.
""")

print(extract_chc(program))
```

yields

```smt
(set-logic HORN)
(declare-fun fact (Int Int) Bool)
(declare-fun fact_acc (Int Int Int) Bool)
(assert (forall ((n Int) (ret Int))
    (=> (fact_acc n 1 ret) (fact n ret))))
(assert (forall ((acc Int) (acc_p Int) (n Int) (n_p Int) (ret Int))
    (=> (and (> n 0) (= acc_p (* acc n)) (= n_p (- n 1))
             (fact_acc n_p acc_p ret))
        (fact_acc n acc ret))))
(assert (forall ((acc Int) (n Int))
    (=> (<= n 0) (fact_acc n acc acc))))
```

To verify a safety property, append a negated-Horn query and ask
Z3's Spacer engine:

```python
import z3

smt = extract_chc(program) + """
(assert (forall ((n Int) (r Int))
    (=> (and (fact n r) (>= n 0) (< r 1)) false)))
"""
s = z3.SolverFor("HORN")
s.from_string(smt)
print(s.check())   # sat  (property holds: fact(n) >= 1 for n >= 0)
```

Spacer's results read inverted from a normal solver:

- `sat`     — property holds (no counterexample exists)
- `unsat`   — property violated (counterexample exists)
- `unknown` — solver gave up (common with `mod` or heavy array reasoning)

### 4. State-passing transform (globals and arrays)

Procedures that touch state have their mutable inputs threaded through
the CHC predicate automatically. The extractor runs a transitive
read/write analysis so each predicate only carries the state it
actually uses. For `heap_push` from `wam_runtime.lp`:

```smt
(declare-fun heap_push (Int Int Int (Array Int Int)   ; tag val H_in HEAP_in
                        Int Int (Array Int Int))      ; addr H_out HEAP_out
              Bool)
```

i32 globals become `Int` pre/post parameters; i32 arrays become
`(Array Int Int)` pre/post parameters with `select`/`store`. Funcref
arrays are abstracted to no-ops (sound but uninformative).

Run all CHC tests:

```bash
python3 test_chc.py
```

### 5. Putting it together: the WAM in LP Form

`wam_runtime.lp` implements the full Warren Abstract Machine runtime
— heap, trail, PDL, deref, bind, unification, backtracking — as LP
Form. Compile and run the built-in `test` procedure (which unifies
`f(X)` with `f(42)` and reads back `X`):

```bash
python3 test_wam_runtime.py
```

The same source can be passed to `chc.py` and fed to Z3 for property
checking (the declarations parse; properties about deref termination
and trail consistency are future work).

## Prolog, as a secondary frontend

The Prolog frontend compiles through the LP Form pipeline:

- `prolog_parser.py` — Lark grammar for a Prolog subset
- `normalize_pass.py` — normalizes surface AST (expands lists, etc.)
- `prolog_to_lp.py` — lowers normalized Prolog to LP Form IR
- `plwasm.py` — `compile(source)` ties it all together: parse →
  normalize → LP Form → link with `wam_runtime.lp` → WASM

The old direct WAM path (`prolog_to_wam.py`, `wam_emit.py`) still
exists but is no longer used by the tests.

## Running the tests

```bash
python3 test_lp.py            # LP Form → WASM end-to-end (9 tests)
python3 test_chc.py           # CHC extraction + Z3 verification (34 tests)
python3 test_types.py         # Type system: structs, sums, ADTs (22 tests)
python3 test_wam_runtime.py   # WAM runtime compiled from LP Form
python3 test_e2e.py           # Prolog → LP Form → WASM end-to-end (7 tests)
python3 test_phase6.py        # Prolog via plwasm.compile (6 tests)
python3 test_trace.py         # Trace instrumentation (6 tests)
python3 test_debugger.py      # Trace-driven debugger (7 tests)
python3 test_nanopass.py      # Nanopass framework unit tests
```

## References

- Gange, Navas, Schachte, Sondergaard, Stuckey. *Horn Clauses as an
  Intermediate Representation for Program Analysis and Transformation.*
  TPLP 2015. — the paper LP Form is based on.
- Aït-Kaci. *Warren's Abstract Machine: A Tutorial Reconstruction.* —
  the reference for the WAM runtime in `wam_runtime.lp`.
- Sarkar, Waddell, Dybvig. *A Nanopass Infrastructure for Compiler
  Education.* ICFP 2004. — inspiration for the Prolog pass tower.
- Komuravelli, Gurfinkel, Chaki. *SMT-Based Model Checking for
  Recursive Programs.* (Spacer) — the CHC engine in Z3 that
  `chc.py` targets.

## License

MIT.
