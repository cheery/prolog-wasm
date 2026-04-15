"""End-to-end test: Prolog -> WAM -> WASM -> run.

Compiles a minimal Prolog program through the full pipeline and
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
    FN_INIT, FN_HEAP_GET_TAG, FN_HEAP_GET_VAL, FN_DEREF,
    FN_BACKTRACK_RESTORE,
    FT_CLAUSE, T_CONT,
    emit_xget, emit_xset,
    TAG_REF, TAG_CON,
    G_H, G_FAIL, G_B, G_BP_TOP, G_BP_STACK,
    build_module as build_runtime_module,
    validate_wasm, run_wasm,
)
from prolog_parser import parse
from prolog_compiler import compile_program, WAMInstruction
from wam_emit import ClauseEmitter
from symbols import SymbolTable


def build_prolog_module(source: str):
    """Compile Prolog source to a WASM module. Returns (wasm_bytes, syms)."""

    # 1. Parse and compile to WAM
    program = parse(source)
    predicates, queries = compile_program(program)

    # 2. Intern symbols
    syms = SymbolTable()
    syms.intern_program(predicates, queries)

    # 3. Start with the runtime module structure
    types = runtime_types()
    func_types = []       # (functype, code_bytes) for each function
    func_type_indices = []
    func_codes = []

    # Add runtime functions (indices 0-11, matching FN_* constants)
    from wam_wasm import (
        build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
        build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
        build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
    )
    runtime_builders = [
        build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
        build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
        build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
    ]
    for builder in runtime_builders:
        ft, code = builder()
        ti = len(types)
        types.append(ft)
        func_type_indices.append(ti)
        func_codes.append(code)

    # Next available function index
    next_fn = len(func_type_indices)

    # 4. Compile each predicate clause to a WASM function
    #    First pass: assign function indices
    func_indices = {}   # "pred/N_cN" or "pred/N" -> WASM function index

    for key, pred in predicates.items():
        first_fn = next_fn
        for ci, (label, instrs) in enumerate(pred.clauses):
            func_indices[label] = next_fn
            next_fn += 1
        # Also register the predicate key ("name/arity") → first clause,
        # so call/execute instructions can look it up by predicate name.
        func_indices[key] = first_fn

    # 5. Second pass: emit WASM bodies using ClauseEmitter
    emitter = ClauseEmitter(syms, func_indices)

    clause_func_indices_ordered = []  # in order of func_indices assignment

    for key, pred in predicates.items():
        for ci, (label, instrs) in enumerate(pred.clauses):
            # Each clause function has type [] -> []
            clause_type = functype([], [])
            clause_ti = len(types)
            types.append(clause_type)
            func_type_indices.append(clause_ti)

            wir = emitter.emit(instrs)
            func_codes.append(wir.encode())
            clause_func_indices_ordered.append(func_indices[label])

    # Collect funcrefs declared via ref_func in try_me_else / retry_me_else
    declared_funcrefs = list(emitter.declared_funcrefs)

    # 6. Add a test function that:
    #    - calls init()
    #    - runs the query (put args in X regs, call predicate)
    #    - reads X1, derefs, reads value, returns it
    target_key = None
    query_instrs = None
    query_reg_map = None

    if queries:
        query_instrs, query_reg_map = queries[0]
        for instr in query_instrs:
            if instr.opcode in ('call', 'execute'):
                name, arity = instr.args[0]
                target_key = f"{name}/{arity}"
                break

    if target_key and target_key in func_indices and query_instrs:
        # Build a test that: init(), run query instructions, return X1
        test_type = functype([], [I32])
        test_ti = len(types)
        types.append(test_type)

        test_ir = WIR([], I32)
        test_ir.new_local('result')
        test_ir.fn_call(FN_INIT)

        # Emit query instructions using the emitter
        # For the simple test, manually emit: put_variable X1, 1; execute pred/N
        # But let's use the emitter for the query too
        query_emitter = ClauseEmitter(syms, func_indices)
        query_wir = query_emitter.emit(query_instrs)

        # But query_wir is for a [] -> [] function. We need to splice its
        # instructions into our test function. That's not directly possible.
        # Instead, make the query a separate function and call it.

        # Add query as a [] -> [] function
        query_clause_type = functype([], [])
        query_clause_ti = len(types)
        types.append(query_clause_type)
        func_type_indices.append(query_clause_ti)
        func_codes.append(query_wir.encode())
        query_fn_idx = len(func_type_indices) - 1

        # Test function: init(); call query_fn; backtrack loop; read X1
        test_ir.fn_call(query_fn_idx)

        # Backtracking loop: retry until success or no more choice points
        with test_ir.while_loop() as loop:
            # Break on success
            test_ir.gget(G_FAIL); test_ir.eqz(); loop.break_if()
            # Break if no choice point remains
            test_ir.gget(G_B); test_ir.const(-1); test_ir.eq(); loop.break_if()
            # Restore state from current choice point
            test_ir.fn_call(FN_BACKTRACK_RESTORE)
            # Call the next clause (BP_STACK[G_BP_TOP - 1])
            test_ir.gget(G_BP_STACK)
            test_ir.gget(G_BP_TOP); test_ir.const(1); test_ir.sub()
            test_ir._emit(array_get(T_CONT))
            test_ir.call_ref(FT_CLAUSE)

        # Read X1: deref, get value (works for variable or constant at A1)
        emit_xget(test_ir, 1)
        test_ir.fn_call(FN_DEREF)
        test_ir.fn_call(FN_HEAP_GET_VAL)

        func_type_indices.append(test_ti)
        func_codes.append(test_ir.encode())
        test_fn_idx = len(func_type_indices) - 1
    else:
        # Fallback: just return 0
        test_type = functype([], [I32])
        test_ti = len(types)
        types.append(test_type)
        test_ir = WIR([], I32)
        test_ir.const(0)
        func_type_indices.append(test_ti)
        func_codes.append(test_ir.encode())
        test_fn_idx = len(func_type_indices) - 1
        query_fn_idx = None

    # Build the module
    wasm = module(
        types=types,
        funcs=func_type_indices,
        globals_=runtime_globals(),
        codes=func_codes,
        exports=[
            export_func("test", test_fn_idx),
        ],
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
        expected4 = syms4.encode_constant(42)   # 42 as integer = 42
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
    # We construct the WAM instructions for pick_with_cut/1 by hand so we can
    # exercise get_level + cut without waiting for the compiler to emit them.
    print("\n=== Test 7: get_level / cut — deep cut ===")
    from prolog_compiler import WAMInstruction, compile_program, print_compiled
    from prolog_parser import parse

    src7 = """
        alt(1). alt(2). alt(3).
    """
    program7 = parse(src7)
    predicates7, _ = compile_program(program7)

    syms7 = SymbolTable()
    syms7.intern_program(predicates7, [])

    # Build func_indices for alt/1 clauses
    base_fn = 12   # runtime (0-11)
    fi7 = {}
    next7 = base_fn
    for key, pred in predicates7.items():
        first7 = next7
        for ci, (label, instrs) in enumerate(pred.clauses):
            fi7[label] = next7; next7 += 1
        fi7[key] = first7

    # Manually build WAM instructions for pick_with_cut/1:
    #   allocate 1          ; Y1 = cut level
    #   get_level Y1        ; save current B into Y1
    #   put_variable X2, 1  ; X = fresh var in A1
    #   call alt/1          ; get first alternative
    #   cut Y1              ; deep cut: remove alt/1's choice points
    #   deallocate
    #   proceed
    pwc_instrs = [
        WAMInstruction("allocate",     [1]),
        WAMInstruction("get_level",    ["Y1"]),
        WAMInstruction("put_variable", ["X2", 1]),
        WAMInstruction("call",         [("alt", 1)]),
        WAMInstruction("cut",          ["Y1"]),
        WAMInstruction("deallocate"),
        WAMInstruction("proceed"),
    ]

    # Add pick_with_cut/1 and its query to func_indices
    pwc_fn = next7; next7 += 1
    fi7["pick_with_cut/1"] = pwc_fn
    query7_fn = next7; next7 += 1

    from wam_emit import ClauseEmitter as CE7
    em7 = CE7(syms7, fi7)

    types7 = runtime_types()
    ftidxs7 = []
    fcodes7 = []
    from wam_wasm import (
        build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
        build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
        build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
    )
    for builder in [build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
                    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
                    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore]:
        ft, code = builder()
        types7.append(ft); ftidxs7.append(len(types7)-1); fcodes7.append(code)

    # Emit alt/1 clause bodies
    for key, pred in predicates7.items():
        for ci, (label, instrs) in enumerate(pred.clauses):
            ct = functype([], []); cti = len(types7); types7.append(ct)
            ftidxs7.append(cti); fcodes7.append(em7.emit(instrs).encode())

    # Emit pick_with_cut/1
    pct = functype([], []); pcti = len(types7); types7.append(pct)
    ftidxs7.append(pcti); fcodes7.append(em7.emit(pwc_instrs).encode())

    # Query: put_variable X2, 1; execute pick_with_cut/1
    q7_instrs = [
        WAMInstruction("put_variable", ["X2", 1]),
        WAMInstruction("execute",      [("pick_with_cut", 1)]),
    ]
    qt = functype([], []); qti = len(types7); types7.append(qt)
    ftidxs7.append(qti); fcodes7.append(em7.emit(q7_instrs).encode())

    # Test driver
    tt = functype([], [I32]); tti = len(types7); types7.append(tt)
    tir = WIR([], I32)
    tir.fn_call(FN_INIT)
    tir.fn_call(query7_fn)
    with tir.while_loop() as loop:
        tir.gget(G_FAIL); tir.eqz(); loop.break_if()
        tir.gget(G_B); tir.const(-1); tir.eq(); loop.break_if()
        tir.fn_call(FN_BACKTRACK_RESTORE)
        tir.gget(G_BP_STACK)
        tir.gget(G_BP_TOP); tir.const(1); tir.sub()
        tir._emit(array_get(T_CONT))
        tir.call_ref(FT_CLAUSE)
    emit_xget(tir, 1); tir.fn_call(FN_DEREF); tir.fn_call(FN_HEAP_GET_VAL)
    test7_fn = len(ftidxs7)
    ftidxs7.append(tti); fcodes7.append(tir.encode())

    decl7 = list(em7.declared_funcrefs)
    wasm7 = module(
        types=types7, funcs=ftidxs7, globals_=runtime_globals(),
        codes=fcodes7,
        exports=[export_func("test", test7_fn)],
        elements=[elem_declare(decl7)],
    )
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
