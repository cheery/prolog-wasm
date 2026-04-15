"""plwasm — Prolog to WASM compiler pipeline.

Full pipeline: Prolog source → WAM → WASM module bytes.

The emitted module exports:
    run_get(n: i32) -> i32
        Calls init(), runs the first query in the source,
        drives the backtracking loop until success (or no more alternatives),
        then returns heap_get_val(deref(xreg[n])).

Usage:
    from plwasm import compile
    from wam_wasm import run_wasm

    wasm_bytes, syms = compile(source)
    result = run_wasm(wasm_bytes, "run_get", 1)   # read X1 after query
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
    FT_CLAUSE, T_CONT, T_XREG,
    G_FAIL, G_B, G_BP_TOP, G_BP_STACK, G_XREG,
    build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
)
from prolog_parser import parse
from prolog_compiler import compile_program
from wam_emit import ClauseEmitter
from symbols import SymbolTable


_RUNTIME_BUILDERS = [
    build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
]


def compile(source: str) -> tuple[bytes, SymbolTable]:
    """Compile Prolog source to a WASM module.

    Returns (wasm_bytes, syms).  The module exports run_get(n: i32) -> i32.
    """
    # 1. Parse and compile to WAM
    program = parse(source)
    predicates, queries = compile_program(program)

    # 2. Intern symbols
    syms = SymbolTable()
    syms.intern_program(predicates, queries)

    # 3. Build type / function tables starting with the runtime
    types = runtime_types()
    func_type_indices = []
    func_codes = []

    for builder in _RUNTIME_BUILDERS:
        ft, code = builder()
        ti = len(types)
        types.append(ft)
        func_type_indices.append(ti)
        func_codes.append(code)

    # 4. First pass: assign function indices to every clause
    func_indices = {}   # "pred/N_cK" and "pred/N" -> WASM function index
    next_fn = len(func_type_indices)

    for key, pred in predicates.items():
        first_fn = next_fn
        for ci, (label, _instrs) in enumerate(pred.clauses):
            func_indices[label] = next_fn
            next_fn += 1
        func_indices[key] = first_fn

    # 5. Second pass: emit WASM clause bodies
    emitter = ClauseEmitter(syms, func_indices)

    for _key, pred in predicates.items():
        for _ci, (label, instrs) in enumerate(pred.clauses):
            clause_ti = len(types)
            types.append(functype([], []))
            func_type_indices.append(clause_ti)
            func_codes.append(emitter.emit(instrs).encode())

    declared_funcrefs = list(emitter.declared_funcrefs)

    # 6. Compile the first query into a separate [] -> [] function
    query_fn_idx = None
    if queries:
        query_instrs, _query_reg_map = queries[0]
        q_emitter = ClauseEmitter(syms, func_indices)
        q_wir = q_emitter.emit(query_instrs)
        declared_funcrefs.extend(q_emitter.declared_funcrefs)

        q_ti = len(types)
        types.append(functype([], []))
        func_type_indices.append(q_ti)
        func_codes.append(q_wir.encode())
        query_fn_idx = len(func_type_indices) - 1

    # 7. Build run_get(n: i32) -> i32
    #    init(); call query; backtrack loop; return heap_val(deref(xreg[n]))
    rg_ti = len(types)
    types.append(functype([I32], [I32]))

    # 'n' is the parameter (local index 0): which X register to return
    rg = WIR(['n'], I32)
    rg.fn_call(FN_INIT)

    if query_fn_idx is not None:
        rg.fn_call(query_fn_idx)

        # Backtracking loop: retry until success or no more choice points
        with rg.while_loop() as loop:
            rg.gget(G_FAIL); rg.eqz(); loop.break_if()           # success
            rg.gget(G_B); rg.const(-1); rg.eq(); loop.break_if() # no more CPs
            rg.fn_call(FN_BACKTRACK_RESTORE)
            rg.gget(G_BP_STACK)
            rg.gget(G_BP_TOP); rg.const(1); rg.sub()
            rg._emit(array_get(T_CONT))
            rg.call_ref(FT_CLAUSE)

    # Read xreg[n] at runtime, deref, return heap value
    rg.gget(G_XREG); rg.local('n'); rg._emit(array_get(T_XREG))
    rg.fn_call(FN_DEREF)
    rg.fn_call(FN_HEAP_GET_VAL)

    rg_fn_idx = len(func_type_indices)
    func_type_indices.append(rg_ti)
    func_codes.append(rg.encode())

    # 8. Assemble module
    wasm = module(
        types=types,
        funcs=func_type_indices,
        globals_=runtime_globals(),
        codes=func_codes,
        exports=[export_func("run_get", rg_fn_idx)],
        elements=[elem_declare(declared_funcrefs)],
    )
    return wasm, syms
