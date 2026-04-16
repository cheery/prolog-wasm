"""End-to-end test: Prolog -> WAM -> WASM -> run.

Compiles a minimal Prolog program through the full nanopass pipeline and
validates it with wasmtime.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import (
    module, functype, export_func, elem_declare, I32, array_get,
)
from wir import WIR
from wam_wasm import (
    runtime_types, runtime_globals,
    FN_INIT, FN_HEAP_GET_VAL, FN_DEREF,
    FN_BACKTRACK_RESTORE,
    FT_CLAUSE, T_CONT,
    emit_xget,
    G_FAIL, G_B, G_BP_TOP, G_BP_STACK,
    build_module as build_runtime_module,
    validate_wasm, run_wasm,
    build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
)
from prolog_parser import parse
from normalize_pass import NormalizeLists
from prolog_to_wam import compile_program_l2
from symbols import SymbolTable, build_func_indices, InternSymbols
from wam_emit import ClauseEmitterL3
from languages import L2


_RUNTIME_BUILDERS = [
    build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
]


def build_prolog_module(source: str):
    """Compile Prolog source to a WASM module. Returns (wasm_bytes, syms)."""
    program_l0 = parse(source)
    program_l1 = NormalizeLists()(program_l0)
    program_l2 = compile_program_l2(program_l1)
    return _compile_l2_program(program_l2)


def _compile_l2_program(program_l2):
    """Lower an L2 Program2 all the way to WASM bytes.

    Returns (wasm_bytes, syms).  The module exports a 'test' function
    that runs the first query and returns the value of X1.
    """
    # 1. Add runtime functions (indices 0-11)
    types = runtime_types()
    func_type_indices = []
    func_codes = []

    for builder in _RUNTIME_BUILDERS:
        ft, code = builder()
        ti = len(types)
        types.append(ft)
        func_type_indices.append(ti)
        func_codes.append(code)

    # 2. Assign WASM function indices to every clause
    fi = build_func_indices(program_l2, base_fn=len(func_type_indices))

    # 3. Resolve symbols: L2 -> L3
    syms = SymbolTable()
    syms.intern_predicate_names(program_l2)
    program_l3 = InternSymbols(syms, fi)(program_l2)

    # 4. Emit clause bodies
    declared_funcrefs = []
    emitter = ClauseEmitterL3(declared_funcrefs)

    for pred in program_l3.predicates:
        for clause in pred.clauses:
            clause_ti = len(types)
            types.append(functype([], []))
            func_type_indices.append(clause_ti)
            func_codes.append(emitter.emit(clause.instrs).encode())

    # 5. Emit first query as a [] -> [] function
    query_fn_idx = None
    if program_l3.queries:
        q = program_l3.queries[0]
        q_emitter = ClauseEmitterL3(declared_funcrefs)
        q_ti = len(types)
        types.append(functype([], []))
        func_type_indices.append(q_ti)
        func_codes.append(q_emitter.emit(q.instrs).encode())
        query_fn_idx = len(func_type_indices) - 1

    # 6. Build test() -> i32
    test_ti = len(types)
    types.append(functype([], [I32]))

    test_ir = WIR([], I32)
    test_ir.fn_call(FN_INIT)

    if query_fn_idx is not None:
        test_ir.fn_call(query_fn_idx)

        with test_ir.while_loop() as loop:
            test_ir.gget(G_FAIL); test_ir.eqz(); loop.break_if()
            test_ir.gget(G_B); test_ir.const(-1); test_ir.eq(); loop.break_if()
            test_ir.fn_call(FN_BACKTRACK_RESTORE)
            test_ir.gget(G_BP_STACK)
            test_ir.gget(G_BP_TOP); test_ir.const(1); test_ir.sub()
            test_ir._emit(array_get(T_CONT))
            test_ir.call_ref(FT_CLAUSE)

    emit_xget(test_ir, 1)
    test_ir.fn_call(FN_DEREF)
    test_ir.fn_call(FN_HEAP_GET_VAL)

    test_fn_idx = len(func_type_indices)
    func_type_indices.append(test_ti)
    func_codes.append(test_ir.encode())

    # 7. Assemble module
    wasm = module(
        types=types,
        funcs=func_type_indices,
        globals_=runtime_globals(),
        codes=func_codes,
        exports=[export_func("test", test_fn_idx)],
        elements=[elem_declare(declared_funcrefs)],
    )
    return wasm, syms


if __name__ == "__main__":
    # Test 1: Simple fact (validation only — no query)
    source = """
        parent(tom, bob).
    """
    print("=== Test 1: single fact parent(tom, bob) ===")
    wasm, _ = build_prolog_module(source)
    with open("stage3.wasm", "wb") as f:
        f.write(wasm)
    print(f"Module: {len(wasm)} bytes")
    ok = validate_wasm(wasm, "single fact")
    if ok:
        print("  Validation passed!")

    # Test 2: Fact with a number + query — full end-to-end execution
    source2 = """
        answer(42).
        ?- answer(X).
    """
    print("\n=== Test 2: answer(42) + query ?- answer(X). ===")
    wasm2, _ = build_prolog_module(source2)
    with open("stage3b.wasm", "wb") as f:
        f.write(wasm2)
    print(f"Module: {len(wasm2)} bytes")
    ok = validate_wasm(wasm2, "answer(42) + query")
    if ok:
        print("  Validation passed!")
        result = run_wasm(wasm2, "test")
        print(f"  test() = {result}  (expected 42)")
        assert result == 42, f"Expected 42, got {result}"
        print("  PASS")

    # Test 3: Atom constant fact + query
    source3 = """
        greeting(hello).
        ?- greeting(X).
    """
    print("\n=== Test 3: greeting(hello) + query ?- greeting(X). ===")
    wasm3, syms3 = build_prolog_module(source3)
    ok = validate_wasm(wasm3, "greeting(hello) + query")
    if ok:
        print("  Validation passed!")
        result3 = run_wasm(wasm3, "test")
        hello_id = syms3.encode_constant("hello")
        print(f"  test() = {result3}  (expected hello atom id = {hello_id})")
        assert result3 == hello_id, f"Expected {hello_id}, got {result3}"
        print("  PASS")

    # Test 4: Backtracking — first clause fails, second succeeds
    source4 = """
        result(bad).
        result(42).
        ?- result(42).
    """
    print("\n=== Test 4: backtracking — result(bad); result(42); ?- result(42). ===")
    wasm4, syms4 = build_prolog_module(source4)
    ok = validate_wasm(wasm4, "backtracking result(42)")
    if ok:
        print("  Validation passed!")
        result4 = run_wasm(wasm4, "test")
        expected4 = syms4.encode_constant(42)
        print(f"  test() = {result4}  (expected {expected4})")
        assert result4 == expected4, f"Expected {expected4}, got {result4}"
        print("  PASS")

    # Test 6: neck_cut — first matching clause commits, no backtracking
    source6 = """
        first(1).
        first(2).
        first(3).
        only_first(X) :- first(X), !.
        ?- only_first(X).
    """
    print("\n=== Test 6: neck_cut — only_first/1 should return 1, not backtrack ===")
    wasm6, syms6 = build_prolog_module(source6)
    ok = validate_wasm(wasm6, "neck_cut")
    if ok:
        print("  Validation passed!")
        result6 = run_wasm(wasm6, "test")
        print(f"  test() = {result6}  (expected 1)")
        assert result6 == 1, f"Expected 1, got {result6}"
        print("  PASS")

    # Test 7: get_level / cut — deep cut removes choice points across call frames
    #
    # Prolog equivalent:
    #   alt(1). alt(2). alt(3).
    #   pick_with_cut(X) :- get_level(L), alt(X), cut(L).
    #   ?- pick_with_cut(X).
    #
    # We construct the L2 instructions for pick_with_cut/1 by hand so we can
    # exercise get_level + cut without waiting for the compiler to emit them.
    print("\n=== Test 7: get_level / cut — deep cut ===")

    src7 = """
        alt(1). alt(2). alt(3).
    """
    program7_l2 = compile_program_l2(NormalizeLists()(parse(src7)))

    # Hand-build L2 instructions for pick_with_cut/1:
    #   allocate 1        ; reserve Y1 for cut level
    #   get_level Y1      ; save current B
    #   put_variable X2,1 ; X1 = fresh variable
    #   call alt/1        ; bind X1 to first alternative
    #   cut Y1            ; remove alt/1's remaining choice points
    #   deallocate
    #   proceed
    pwc_clause = L2.Clause2(
        label="pick_with_cut/1_c0",
        instrs=[
            L2.Allocate(n=1),
            L2.GetLevel(reg="Y1"),
            L2.PutVariable(reg="X2", ai=1),
            L2.Call(functor="alt", arity=1),
            L2.Cut(reg="Y1"),
            L2.Deallocate(),
            L2.Proceed(),
        ],
    )
    pwc_pred = L2.Predicate2(name="pick_with_cut", arity=1, clauses=[pwc_clause])

    q7 = L2.Query2(
        instrs=[
            L2.PutVariable(reg="X2", ai=1),
            L2.Execute(functor="pick_with_cut", arity=1),
        ],
        reg_map={"X": "X1"},
    )

    combined_l2 = L2.Program2(
        predicates=program7_l2.predicates + [pwc_pred],
        queries=[q7],
    )

    wasm7, syms7 = _compile_l2_program(combined_l2)
    ok = validate_wasm(wasm7, "get_level/cut")
    if ok:
        print("  Validation passed!")
        result7 = run_wasm(wasm7, "test")
        print(f"  test() = {result7}  (expected 1 — cut prevents alt(2), alt(3))")
        assert result7 == 1, f"Expected 1, got {result7}"
        print("  PASS")

    # Test 5: Three-clause backtracking — find the matching atom
    source5 = """
        color(red).
        color(green).
        color(blue).
        ?- color(blue).
    """
    print("\n=== Test 5: three-clause backtrack — color/1 ?- color(blue). ===")
    wasm5, syms5 = build_prolog_module(source5)
    ok = validate_wasm(wasm5, "three-clause backtrack")
    if ok:
        print("  Validation passed!")
        result5 = run_wasm(wasm5, "test")
        blue_id = syms5.encode_constant("blue")
        print(f"  test() = {result5}  (expected blue atom id = {blue_id})")
        assert result5 == blue_id, f"Expected {blue_id}, got {result5}"
        print("  PASS")
