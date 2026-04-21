"""plwasm — Prolog to WASM compiler pipeline.

Full pipeline: Prolog source → normalize → LP Form → link with WAM runtime → WASM.

The emitted module exports:
    run_get(n: i32) -> i32
        Calls init(), runs the first query in the source,
        then returns heap_get_val(deref(xreg[n])).

Usage:
    from plwasm import compile
    from wam_wasm import run_wasm

    wasm_bytes, syms = compile(source)
    result = run_wasm(wasm_bytes, "run_get", 1)
"""
import os

from prolog_parser import parse
from normalize_pass import NormalizeLists
from prolog_to_lp import compile_prolog
from lp_parser import parse_lp
from lp_pipeline import lp_compile


def compile(source: str) -> tuple[bytes, 'SymbolTable']:
    """Compile Prolog source to a WASM module.

    Returns (wasm_bytes, syms).  The module exports run_get(n: i32) -> i32.

    Pipeline:
      parse -> L0 -> NormalizeLists -> L1 -> compile_prolog -> LPProgram
           -> link with wam_runtime.lp -> lp_compile -> WASM bytes
    """
    # 1. Parse: text -> L0
    program_l0 = parse(source)

    # 2. Normalize: L0 -> L1
    program_l1 = NormalizeLists()(program_l0)

    # 3. Lower: L1 -> LP Form
    user_program, syms = compile_prolog(program_l1)

    # 4. Parse WAM runtime
    runtime_path = os.path.join(os.path.dirname(__file__), 'wam_runtime.lp')
    with open(runtime_path) as f:
        runtime_source = f.read()
    runtime = parse_lp(runtime_source)
    runtime.entry = None  # export all procs by name, no entry wrapper

    # 5. Link runtime + user program
    linked = runtime.link(user_program)
    linked.entry = None

    # 6. Compile to WASM
    wasm_bytes = lp_compile(linked)

    return wasm_bytes, syms
