"""LP Form compilation pipeline.

Validates, elaborates, and compiles LP Form programs to WASM modules.

Usage:
    from lp_form import *
    from lp_pipeline import lp_compile

    program = LPProgram(procedures=[...], entry="main")
    wasm_bytes = lp_compile(program)
"""

from lp_form import validate, mark_tail_calls, infer_output_types, LPProgram
from lp_elaborate import elaborate
from lp_emit import LPEmitter


def lp_compile(program: LPProgram, trace: bool = False,
               trace_size: int = 1 << 20) -> bytes:
    """Compile an LP Form program to WASM module bytes.

    1. Validates the program structure
    2. Infers output types for ref-typed returns (before elaboration,
       so elaboration can resolve field access on call outputs)
    3. Elaborates type features (patterns, field access)
    4. Marks tail calls for optimization
    5. Emits WASM via LPEmitter

    When trace=True, the emitter also writes a per-clause execution trace
    to an i32 GC array and exports accessor functions (see LPEmitter docs).
    """
    validate(program)
    infer_output_types(program)

    # Run elaboration if there are type declarations or patterns/field access
    needs_elaboration = (
        program.structs or program.sums or _has_patterns(program))
    if needs_elaboration:
        program = elaborate(program)

    mark_tail_calls(program)
    return LPEmitter(trace=trace, trace_size=trace_size).compile(program)


def _has_patterns(program):
    """Check if any clause has LPPattern or LPFieldAccess nodes."""
    from lp_form import LPPattern, LPFieldAccess, PrimOp, Call, Guard
    for proc in program.procedures:
        for clause in proc.clauses:
            for goal in clause.goals:
                if isinstance(goal, (PrimOp, Call)):
                    for out in goal.outputs:
                        if isinstance(out, LPPattern):
                            return True
                    for inp in goal.inputs:
                        if isinstance(inp, LPFieldAccess):
                            return True
                if isinstance(goal, Guard):
                    if isinstance(goal.left, LPFieldAccess):
                        return True
                    if isinstance(goal.right, LPFieldAccess):
                        return True
    return False
