"""WAM-to-WASM emitter: translates WAM instructions to WASM function bodies.

Each compiled WAM clause becomes a WASM function with signature [] -> [].
The emitter uses the runtime helpers from wam_wasm.py and emits inline
code for registers, globals, and control flow.

Covers Phase 3 (data instructions) and Phase 4 (environments/calls).
Phase 5 (choice points) will be added later.

IMPORTANT: WIR control flow (if_, if_else, while_loop) consumes the
condition from the stack in __enter__.  So the condition MUST be pushed
BEFORE the `with` statement.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import array_get, array_set
from wir import WIR
from wam_wasm import (
    TAG_REF, TAG_CON, TAG_STR, TAG_LIS, TAG_FUN,
    G_H, G_E, G_B, G_S, G_MODE, G_FAIL, G_HB, G_TR, G_ESTACK,
    G_TRAIL, G_STACK, G_BP_TOP, G_BP_STACK,
    T_HEAP, T_TRAIL, T_STACK, T_XREG, T_CONT,
    FN_HEAP_PUSH, FN_HEAP_GET_TAG, FN_HEAP_GET_VAL,
    FN_HEAP_SET_TAG, FN_HEAP_SET_VAL,
    FN_DEREF, FN_BIND, FN_PDL_PUSH, FN_PDL_POP, FN_UNIFY,
    FT_CLAUSE,
    emit_xget, emit_xset, emit_cont_push, emit_cont_pop,
)
MODE_WRITE = 0
MODE_READ  = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_reg(vn: str) -> tuple[str, int]:
    return vn[0], int(vn[1:])


def _reg_get(ir: WIR, vn: str):
    """Push value of register Xn or Yn onto WASM stack."""
    kind, idx = _parse_reg(vn)
    if kind == 'X':
        emit_xget(ir, idx)
    else:
        ir.gget(G_STACK); ir.gget(G_E); ir.const(idx); ir.add()
        ir._emit(array_get(T_STACK))


def _reg_set(ir: WIR, vn: str, emit_value):
    """Set register Xn or Yn. emit_value() must push the value."""
    kind, idx = _parse_reg(vn)
    if kind == 'X':
        emit_xset(ir, idx, emit_value)
    else:
        ir.gget(G_STACK)
        ir.gget(G_E); ir.const(idx); ir.add()
        emit_value()
        ir._emit(array_set(T_STACK))


def _trail(ir: WIR, addr_local: str):
    """Inline trail: if addr < HB, trail[TR] = addr; TR++."""
    ir.local(addr_local); ir.gget(G_HB); ir.lt_s()
    with ir.if_():
        ir.gget(G_TRAIL); ir.gget(G_TR); ir.local(addr_local)
        ir._emit(array_set(T_TRAIL))
        ir.gget(G_TR); ir.const(1); ir.add(); ir.gset(G_TR)


# ===========================================================================
# ClauseEmitterL3  (nanopass pipeline)
# ===========================================================================

from dataclasses import dataclass as _dataclass
from languages import L3 as _L3


@_dataclass
class CompiledModule:
    """Intermediate result of EmitWASM: everything needed to assemble the
    final WASM module, but not yet encoded to bytes."""
    types: list               # list of functype values
    func_type_indices: list   # list of type indices, one per function
    func_codes: list          # list of encoded function bodies (bytes)
    declared_funcrefs: list   # function indices used via ref_func
    query_fn_idx: int | None  # WASM function index of the query function


class ClauseEmitterL3:
    """Emits a WASM function body for one L3 clause.

    Identical in structure to ClauseEmitter but dispatches on L3 typed
    nodes (using type(instr).__name__) rather than WAMInstruction.opcode.

    Usage:
        em = ClauseEmitterL3(declared_funcrefs_list)
        wir = em.emit(l3_clause.instrs)
        body_bytes = wir.encode()
    """

    def __init__(self, declared_funcrefs: list):
        self.declared_funcrefs = declared_funcrefs

    def emit(self, instrs: list) -> WIR:
        ir = WIR([])
        ir.new_local('a')
        ir.new_local('b')

        for instr in instrs:
            handler = getattr(self, f'_op_{type(instr).__name__}', None)
            if handler is None:
                raise ValueError(
                    f"ClauseEmitterL3: no handler for {type(instr).__name__}"
                )
            handler(ir, instr)

        return ir

    # ------------------------------------------------------------------
    # PUT instructions
    # ------------------------------------------------------------------

    def _op_PutVariable(self, ir, instr):
        ir.const(TAG_REF); ir.gget(G_H); ir.fn_call(FN_HEAP_PUSH)
        ir.set('a')
        _reg_set(ir, instr.reg, lambda: ir.local('a'))
        emit_xset(ir, instr.ai, lambda: ir.local('a'))

    def _op_PutValue(self, ir, instr):
        emit_xset(ir, instr.ai, lambda: _reg_get(ir, instr.reg))

    def _op_PutUnsafeValue(self, ir, instr):
        # Same as PutValue for now (full globalizing not yet implemented)
        emit_xset(ir, instr.ai, lambda: _reg_get(ir, instr.reg))

    def _op_PutStructure(self, ir, instr):
        packed = instr.functor_packed
        ir.const(TAG_STR)
        ir.gget(G_H); ir.const(1); ir.add()
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        ir.const(TAG_FUN); ir.const(packed)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()
        emit_xset(ir, instr.ai, lambda: ir.local('a'))
        ir.const(MODE_WRITE); ir.gset(G_MODE)

    def _op_PutList(self, ir, instr):
        ir.const(TAG_LIS)
        ir.gget(G_H); ir.const(1); ir.add()
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        ir.gget(G_H); ir.gset(G_S)
        emit_xset(ir, instr.ai, lambda: ir.local('a'))
        ir.const(MODE_WRITE); ir.gset(G_MODE)

    def _op_PutConstant(self, ir, instr):
        ir.const(TAG_CON); ir.const(instr.value)
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        emit_xset(ir, instr.ai, lambda: ir.local('a'))

    # ------------------------------------------------------------------
    # GET instructions
    # ------------------------------------------------------------------

    def _op_GetVariable(self, ir, instr):
        _reg_set(ir, instr.reg, lambda: emit_xget(ir, instr.ai))

    def _op_GetValue(self, ir, instr):
        _reg_get(ir, instr.reg)
        emit_xget(ir, instr.ai)
        ir.fn_call(FN_UNIFY)

    def _op_GetStructure(self, ir, instr):
        packed = instr.functor_packed
        ir.new_local('gs_addr'); ir.new_local('gs_tag'); ir.new_local('gs_str')

        emit_xget(ir, instr.ai); ir.fn_call(FN_DEREF); ir.set('gs_addr')
        ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gs_tag')

        ir.local('gs_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            ir.const(TAG_STR)
            ir.gget(G_H); ir.const(1); ir.add()
            ir.fn_call(FN_HEAP_PUSH); ir.set('gs_str')
            ir.const(TAG_FUN); ir.const(packed)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()
            ir.local('gs_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gs_addr'); ir.local('gs_str'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gs_addr')
            ir.const(MODE_WRITE); ir.gset(G_MODE)

            ie.then_part()
            ir.local('gs_tag'); ir.const(TAG_STR); ir.eq()
            with ir.if_else() as ie2:
                ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(packed); ir.eq()
                ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(1); ir.add(); ir.gset(G_S)
                ir.const(MODE_READ); ir.gset(G_MODE)
                ie2.then_part()
                ir.const(1); ir.gset(G_FAIL)

            ir.local('gs_tag'); ir.const(TAG_REF); ir.ne()
            ir.local('gs_tag'); ir.const(TAG_STR); ir.ne()
            ir.and_()
            with ir.if_():
                ir.const(1); ir.gset(G_FAIL)

    def _op_GetList(self, ir, instr):
        ir.new_local('gl_addr'); ir.new_local('gl_tag'); ir.new_local('gl_lis')

        emit_xget(ir, instr.ai); ir.fn_call(FN_DEREF); ir.set('gl_addr')
        ir.local('gl_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gl_tag')

        ir.local('gl_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            ir.const(TAG_LIS)
            ir.gget(G_H); ir.const(1); ir.add()
            ir.fn_call(FN_HEAP_PUSH); ir.set('gl_lis')
            ir.local('gl_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gl_addr'); ir.local('gl_lis'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gl_addr')
            ir.gget(G_H); ir.gset(G_S)
            ir.const(MODE_WRITE); ir.gset(G_MODE)

            ie.then_part()
            ir.local('gl_tag'); ir.const(TAG_LIS); ir.eq()
            with ir.if_else() as ie2:
                ir.local('gl_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.gset(G_S)
                ir.const(MODE_READ); ir.gset(G_MODE)
                ie2.then_part()
                ir.const(1); ir.gset(G_FAIL)

    def _op_GetConstant(self, ir, instr):
        val = instr.value
        ir.new_local('gc_addr'); ir.new_local('gc_tag'); ir.new_local('gc_c')

        emit_xget(ir, instr.ai); ir.fn_call(FN_DEREF); ir.set('gc_addr')
        ir.local('gc_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gc_tag')

        ir.local('gc_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            ir.const(TAG_CON); ir.const(val)
            ir.fn_call(FN_HEAP_PUSH); ir.set('gc_c')
            ir.local('gc_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gc_addr'); ir.local('gc_c'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gc_addr')

            ie.then_part()
            ir.local('gc_tag'); ir.const(TAG_CON); ir.eq()
            with ir.if_():
                ir.local('gc_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(val); ir.ne()
                with ir.if_():
                    ir.const(1); ir.gset(G_FAIL)
            ir.local('gc_tag'); ir.const(TAG_REF); ir.ne()
            ir.local('gc_tag'); ir.const(TAG_CON); ir.ne()
            ir.and_()
            with ir.if_():
                ir.const(1); ir.gset(G_FAIL)

    # ------------------------------------------------------------------
    # UNIFY instructions
    # ------------------------------------------------------------------

    def _op_UnifyVariable(self, ir, instr):
        ir.new_local('uv')

        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            _reg_set(ir, instr.reg, lambda: ir.gget(G_S))
            ir.const(TAG_REF); ir.gget(G_H)
            ir.fn_call(FN_HEAP_PUSH); ir.set('uv')
            _reg_set(ir, instr.reg, lambda: ir.local('uv'))

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_UnifyValue(self, ir, instr):
        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            _reg_get(ir, instr.reg); ir.gget(G_S); ir.fn_call(FN_UNIFY)
            _reg_get(ir, instr.reg); ir.fn_call(FN_HEAP_GET_TAG)
            _reg_get(ir, instr.reg); ir.fn_call(FN_HEAP_GET_VAL)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_UnifyLocalValue(self, ir, instr):
        # Same as UnifyValue for now
        self._op_UnifyValue(ir, instr)

    def _op_UnifyConstant(self, ir, instr):
        val = instr.value
        ir.new_local('uc_addr'); ir.new_local('uc_c')

        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            ir.gget(G_S); ir.fn_call(FN_DEREF); ir.set('uc_addr')
            ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_TAG)
            ir.const(TAG_REF); ir.eq()
            with ir.if_else() as ie2:
                ir.const(TAG_CON); ir.const(val)
                ir.fn_call(FN_HEAP_PUSH); ir.set('uc_c')
                ir.local('uc_addr'); ir.const(TAG_REF)
                ir.fn_call(FN_HEAP_SET_TAG)
                ir.local('uc_addr'); ir.local('uc_c')
                ir.fn_call(FN_HEAP_SET_VAL)
                _trail(ir, 'uc_addr')
                ie2.then_part()
                ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_TAG)
                ir.const(TAG_CON); ir.eq()
                with ir.if_():
                    ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                    ir.const(val); ir.ne()
                    with ir.if_():
                        ir.const(1); ir.gset(G_FAIL)

            ir.const(TAG_CON); ir.const(val)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_UnifyVoid(self, ir, instr):
        for _ in range(instr.n):
            ir.gget(G_MODE); ir.const(MODE_WRITE); ir.eq()
            with ir.if_():
                ir.const(TAG_REF); ir.gget(G_H)
                ir.fn_call(FN_HEAP_PUSH); ir.drop()
            ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    # ------------------------------------------------------------------
    # SET instructions
    # ------------------------------------------------------------------

    def _op_SetVariable(self, ir, instr):
        ir.new_local('sv')
        ir.const(TAG_REF); ir.gget(G_H)
        ir.fn_call(FN_HEAP_PUSH); ir.set('sv')
        _reg_set(ir, instr.reg, lambda: ir.local('sv'))

    def _op_SetValue(self, ir, instr):
        _reg_get(ir, instr.reg); ir.fn_call(FN_HEAP_GET_TAG)
        _reg_get(ir, instr.reg); ir.fn_call(FN_HEAP_GET_VAL)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()

    def _op_SetLocalValue(self, ir, instr):
        self._op_SetValue(ir, instr)

    def _op_SetConstant(self, ir, instr):
        ir.const(TAG_CON); ir.const(instr.value)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()

    def _op_SetVoid(self, ir, instr):
        for _ in range(instr.n):
            ir.const(TAG_REF); ir.gget(G_H)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

    # ------------------------------------------------------------------
    # Control instructions
    # ------------------------------------------------------------------

    def _op_Allocate(self, ir, instr):
        n = instr.n
        ir.gget(G_STACK); ir.gget(G_ESTACK); ir.gget(G_E)
        ir._emit(array_set(T_STACK))
        for i in range(n):
            ir.gget(G_STACK)
            ir.gget(G_ESTACK); ir.const(1 + i); ir.add()
            ir.gget(G_ESTACK); ir.const(1 + i); ir.add()
            ir._emit(array_set(T_STACK))
        ir.gget(G_ESTACK); ir.gset(G_E)
        ir.gget(G_ESTACK); ir.const(1 + n); ir.add(); ir.gset(G_ESTACK)

    def _op_Deallocate(self, ir, instr):
        ir.new_local('old_e')
        ir.gget(G_STACK); ir.gget(G_E); ir._emit(array_get(T_STACK))
        ir.set('old_e')
        ir.gget(G_E); ir.gset(G_ESTACK)
        ir.local('old_e'); ir.gset(G_E)

    def _op_Call(self, ir, instr):
        target = instr.func_index
        ir.gget(G_FAIL)
        with ir.if_():
            ir.ret()
        ir.fn_call(target)
        ir.gget(G_FAIL)
        with ir.if_():
            ir.ret()

    def _op_Execute(self, ir, instr):
        ir.return_call(instr.func_index)

    def _op_Proceed(self, ir, instr):
        ir.ret()

    # ------------------------------------------------------------------
    # Choice instructions
    # ------------------------------------------------------------------

    def _op_TryMeElse(self, ir, instr):
        next_func_idx = instr.next_func_index
        n = instr.arity
        self.declared_funcrefs.append(next_func_idx)

        ir.new_local('tme_b')
        ir.gget(G_ESTACK); ir.set('tme_b')

        ir.gget(G_STACK); ir.local('tme_b')
        ir.gget(G_E); ir._emit(array_set(T_STACK))
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(1); ir.add()
        ir.gget(G_B); ir._emit(array_set(T_STACK))
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(2); ir.add()
        ir.gget(G_TR); ir._emit(array_set(T_STACK))
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(3); ir.add()
        ir.gget(G_H); ir._emit(array_set(T_STACK))
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(4); ir.add()
        ir.const(n); ir._emit(array_set(T_STACK))
        for i in range(1, n + 1):
            ir.gget(G_STACK); ir.local('tme_b'); ir.const(4 + i); ir.add()
            emit_xget(ir, i)
            ir._emit(array_set(T_STACK))

        ir.gget(G_BP_STACK); ir.gget(G_BP_TOP)
        ir.ref_func(next_func_idx)
        ir._emit(array_set(T_CONT))
        ir.gget(G_BP_TOP); ir.const(1); ir.add(); ir.gset(G_BP_TOP)

        ir.local('tme_b'); ir.gset(G_B)
        ir.local('tme_b'); ir.const(5 + n); ir.add(); ir.gset(G_ESTACK)
        ir.gget(G_H); ir.gset(G_HB)

    def _op_RetryMeElse(self, ir, instr):
        next_func_idx = instr.next_func_index
        self.declared_funcrefs.append(next_func_idx)

        ir.gget(G_BP_STACK)
        ir.gget(G_BP_TOP); ir.const(1); ir.sub()
        ir.ref_func(next_func_idx)
        ir._emit(array_set(T_CONT))

    def _op_TrustMe(self, ir, instr):
        ir.new_local('tm_old_b')
        ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
        ir._emit(array_get(T_STACK))
        ir.set('tm_old_b')
        ir.local('tm_old_b'); ir.gset(G_B)
        ir.gget(G_BP_TOP); ir.const(1); ir.sub(); ir.gset(G_BP_TOP)

    # ------------------------------------------------------------------
    # Cut instructions
    # ------------------------------------------------------------------

    def _op_NeckCut(self, ir, instr):
        ir.new_local('nc_level')
        ir.gget(G_B); ir.const(-1); ir.ne()
        with ir.if_():
            ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
            ir._emit(array_get(T_STACK))
            ir.set('nc_level')
            self._emit_cut_to(ir, 'nc_level')

    def _op_GetLevel(self, ir, instr):
        _reg_set(ir, instr.reg, lambda: ir.gget(G_B))

    def _op_Cut(self, ir, instr):
        ir.new_local('cut_level')
        _reg_get(ir, instr.reg); ir.set('cut_level')
        self._emit_cut_to(ir, 'cut_level')

    def _emit_cut_to(self, ir, level_local):
        ir.new_local('_ct_old_b')
        with ir.while_loop() as loop:
            ir.gget(G_B); ir.const(-1); ir.eq(); loop.break_if()
            ir.gget(G_B); ir.local(level_local); ir.gt_s()
            ir.eqz(); loop.break_if()

            ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
            ir._emit(array_get(T_STACK))
            ir.set('_ct_old_b')
            ir.local('_ct_old_b'); ir.gset(G_B)
            ir.gget(G_BP_TOP); ir.const(1); ir.sub(); ir.gset(G_BP_TOP)
