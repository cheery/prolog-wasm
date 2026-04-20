"""LP Form compilation pipeline.

Validates, optimizes, and compiles LP Form programs to WASM modules.

Usage:
    from lp_form import *
    from lp_pipeline import lp_compile

    program = LPProgram(procedures=[...], entry="main")
    wasm_bytes = lp_compile(program)
"""

from lp_form import validate, mark_tail_calls, LPProgram
from lp_emit import LPEmitter


def lp_compile(program: LPProgram, trace: bool = False,
               trace_size: int = 1 << 20) -> bytes:
    """Compile an LP Form program to WASM module bytes.

    1. Validates the program structure
    2. Marks tail calls for optimization
    3. Emits WASM via LPEmitter

    When trace=True, the emitter also writes a per-clause execution trace
    to an i32 GC array and exports accessor functions (see LPEmitter docs).
    """
    validate(program)
    mark_tail_calls(program)
    return LPEmitter(trace=trace, trace_size=trace_size).compile(program)
