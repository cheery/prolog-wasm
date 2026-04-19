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


def lp_compile(program: LPProgram) -> bytes:
    """Compile an LP Form program to WASM module bytes.

    1. Validates the program structure
    2. Marks tail calls for optimization
    3. Emits WASM via LPEmitter
    """
    validate(program)
    mark_tail_calls(program)
    return LPEmitter().compile(program)
