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
from normalize_pass import NormalizeLists
from prolog_compiler import compile_program_l2
from wam_emit import ClauseEmitterL3, CompiledModule
from symbols import SymbolTable, build_func_indices, InternSymbols
from languages import L0, L2


_RUNTIME_BUILDERS = [
    build_init, build_heap_push, build_heap_get_tag, build_heap_get_val,
    build_heap_set_tag, build_heap_set_val, build_deref, build_bind,
    build_pdl_push, build_pdl_pop, build_unify, build_backtrack_restore,
]


def compile(source: str) -> tuple[bytes, SymbolTable]:
    """Compile Prolog source to a WASM module.

    Returns (wasm_bytes, syms).  The module exports run_get(n: i32) -> i32.

    Pipeline:
      parse -> L0 -> NormalizeLists -> L1 -> compile_program_l2 -> L2
           -> InternSymbols -> L3 -> ClauseEmitterL3 -> bytes
    """
    # 1. Parse: text -> L0
    program_l0 = parse(source)

    # 2. Normalize: L0 -> L1  (desugar List / BinOp / UnaryOp)
    program_l1 = NormalizeLists()(program_l0)

    # 3. Compile: L1 -> L2  (Prolog terms -> typed WAM instructions)
    program_l2 = compile_program_l2(program_l1)

    # 4. Build WASM type / function tables with the runtime functions
    types = runtime_types()
    func_type_indices = []
    func_codes = []

    for builder in _RUNTIME_BUILDERS:
        ft, code = builder()
        ti = len(types)
        types.append(ft)
        func_type_indices.append(ti)
        func_codes.append(code)

    # 5. Assign WASM function indices to every clause
    fi = build_func_indices(program_l2, base_fn=len(func_type_indices))

    # 6. Resolve symbols: L2 -> L3
    syms = SymbolTable()
    syms.intern_predicate_names(program_l2)
    program_l3 = InternSymbols(syms, fi)(program_l2)

    # 7. Emit WASM clause bodies
    declared_funcrefs = []
    emitter = ClauseEmitterL3(declared_funcrefs)

    for pred in program_l3.predicates:
        for clause in pred.clauses:
            clause_ti = len(types)
            types.append(functype([], []))
            func_type_indices.append(clause_ti)
            func_codes.append(emitter.emit(clause.instrs).encode())

    # 8. Compile the first query into a separate [] -> [] function
    query_fn_idx = None
    if program_l3.queries:
        q = program_l3.queries[0]
        q_emitter = ClauseEmitterL3(declared_funcrefs)
        q_wir = q_emitter.emit(q.instrs)

        q_ti = len(types)
        types.append(functype([], []))
        func_type_indices.append(q_ti)
        func_codes.append(q_wir.encode())
        query_fn_idx = len(func_type_indices) - 1

    # 9. Build run_get(n: i32) -> i32
    rg_ti = len(types)
    types.append(functype([I32], [I32]))

    rg = WIR(['n'], I32)
    rg.fn_call(FN_INIT)

    if query_fn_idx is not None:
        rg.fn_call(query_fn_idx)

        with rg.while_loop() as loop:
            rg.gget(G_FAIL); rg.eqz(); loop.break_if()
            rg.gget(G_B); rg.const(-1); rg.eq(); loop.break_if()
            rg.fn_call(FN_BACKTRACK_RESTORE)
            rg.gget(G_BP_STACK)
            rg.gget(G_BP_TOP); rg.const(1); rg.sub()
            rg._emit(array_get(T_CONT))
            rg.call_ref(FT_CLAUSE)

    rg.gget(G_XREG); rg.local('n'); rg._emit(array_get(T_XREG))
    rg.fn_call(FN_DEREF)
    rg.fn_call(FN_HEAP_GET_VAL)

    rg_fn_idx = len(func_type_indices)
    func_type_indices.append(rg_ti)
    func_codes.append(rg.encode())

    # 10. Assemble module
    wasm = module(
        types=types,
        funcs=func_type_indices,
        globals_=runtime_globals(),
        codes=func_codes,
        exports=[export_func("run_get", rg_fn_idx)],
        elements=[elem_declare(declared_funcrefs)],
    )
    return wasm, syms
