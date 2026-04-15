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
from symbols import SymbolTable

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


# ---------------------------------------------------------------------------
# ClauseEmitter
# ---------------------------------------------------------------------------

class ClauseEmitter:
    """Emits a WASM function body for one WAM clause.

    Usage:
        emitter = ClauseEmitter(syms, func_indices)
        wir = emitter.emit(wam_instructions)
        body_bytes = wir.encode()
    """

    def __init__(self, syms: SymbolTable, func_indices: dict):
        self.syms = syms
        self.func_indices = func_indices
        self.declared_funcrefs = []   # function indices referenced via ref_func

    def emit(self, instrs: list) -> WIR:
        ir = WIR([])  # [] -> []
        ir.new_local('a')
        ir.new_local('b')

        for instr in instrs:
            handler = getattr(self, f'_op_{instr.opcode}', None)
            if handler is None:
                raise ValueError(f"unsupported WAM instruction: {instr.opcode}")
            handler(ir, instr.args)

        return ir

    # ==================================================================
    #  PUT instructions — build terms for body goals
    # ==================================================================

    def _op_put_variable(self, ir, args):
        vn, ai = args
        ir.const(TAG_REF); ir.gget(G_H); ir.fn_call(FN_HEAP_PUSH)
        ir.set('a')
        _reg_set(ir, vn, lambda: ir.local('a'))
        emit_xset(ir, ai, lambda: ir.local('a'))

    def _op_put_value(self, ir, args):
        vn, ai = args
        emit_xset(ir, ai, lambda: _reg_get(ir, vn))

    def _op_put_structure(self, ir, args):
        (name, arity), ai = args
        packed = self.syms.functor_pack(name, arity)
        ir.const(TAG_STR)
        ir.gget(G_H); ir.const(1); ir.add()
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        ir.const(TAG_FUN); ir.const(packed)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()
        emit_xset(ir, ai, lambda: ir.local('a'))
        ir.const(MODE_WRITE); ir.gset(G_MODE)

    def _op_put_list(self, ir, args):
        ai = args[0]
        ir.const(TAG_LIS)
        ir.gget(G_H); ir.const(1); ir.add()
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        ir.gget(G_H); ir.gset(G_S)
        emit_xset(ir, ai, lambda: ir.local('a'))
        ir.const(MODE_WRITE); ir.gset(G_MODE)

    def _op_put_constant(self, ir, args):
        c, ai = args
        val = self.syms.encode_constant(c)
        ir.const(TAG_CON); ir.const(val)
        ir.fn_call(FN_HEAP_PUSH); ir.set('a')
        emit_xset(ir, ai, lambda: ir.local('a'))

    # ==================================================================
    #  GET instructions — match terms in clause head
    # ==================================================================

    def _op_get_variable(self, ir, args):
        vn, ai = args
        _reg_set(ir, vn, lambda: emit_xget(ir, ai))

    def _op_get_value(self, ir, args):
        vn, ai = args
        _reg_get(ir, vn)
        emit_xget(ir, ai)
        ir.fn_call(FN_UNIFY)

    def _op_get_structure(self, ir, args):
        (name, arity), ai = args
        packed = self.syms.functor_pack(name, arity)
        ir.new_local('gs_addr'); ir.new_local('gs_tag'); ir.new_local('gs_str')

        emit_xget(ir, ai); ir.fn_call(FN_DEREF); ir.set('gs_addr')
        ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gs_tag')

        # Condition: tag == REF
        ir.local('gs_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            # REF branch: build structure, bind
            ir.const(TAG_STR)
            ir.gget(G_H); ir.const(1); ir.add()
            ir.fn_call(FN_HEAP_PUSH); ir.set('gs_str')
            ir.const(TAG_FUN); ir.const(packed)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()
            ir.local('gs_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gs_addr'); ir.local('gs_str'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gs_addr')
            ir.const(MODE_WRITE); ir.gset(G_MODE)

            # else: STR branch
            ie.then_part()
            # Condition: tag == STR
            ir.local('gs_tag'); ir.const(TAG_STR); ir.eq()
            with ir.if_else() as ie2:
                # functor match?
                ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(packed); ir.eq()
                # match: set S, read mode
                ir.local('gs_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(1); ir.add(); ir.gset(G_S)
                ir.const(MODE_READ); ir.gset(G_MODE)
                # no match: fail
                ie2.then_part()
                ir.const(1); ir.gset(G_FAIL)

            # neither REF nor STR: fail
            ir.local('gs_tag'); ir.const(TAG_REF); ir.ne()
            ir.local('gs_tag'); ir.const(TAG_STR); ir.ne()
            ir.and_()
            with ir.if_():
                ir.const(1); ir.gset(G_FAIL)

    def _op_get_list(self, ir, args):
        ai = args[0]
        ir.new_local('gl_addr'); ir.new_local('gl_tag'); ir.new_local('gl_lis')

        emit_xget(ir, ai); ir.fn_call(FN_DEREF); ir.set('gl_addr')
        ir.local('gl_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gl_tag')

        ir.local('gl_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            # REF branch: build list, bind
            ir.const(TAG_LIS)
            ir.gget(G_H); ir.const(1); ir.add()
            ir.fn_call(FN_HEAP_PUSH); ir.set('gl_lis')
            ir.local('gl_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gl_addr'); ir.local('gl_lis'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gl_addr')
            ir.gget(G_H); ir.gset(G_S)
            ir.const(MODE_WRITE); ir.gset(G_MODE)

            # else: LIS branch
            ie.then_part()
            ir.local('gl_tag'); ir.const(TAG_LIS); ir.eq()
            with ir.if_else() as ie2:
                ir.local('gl_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.gset(G_S)
                ir.const(MODE_READ); ir.gset(G_MODE)
                ie2.then_part()
                ir.const(1); ir.gset(G_FAIL)

    def _op_get_constant(self, ir, args):
        c, ai = args
        val = self.syms.encode_constant(c)
        ir.new_local('gc_addr'); ir.new_local('gc_tag'); ir.new_local('gc_c')

        emit_xget(ir, ai); ir.fn_call(FN_DEREF); ir.set('gc_addr')
        ir.local('gc_addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('gc_tag')

        ir.local('gc_tag'); ir.const(TAG_REF); ir.eq()
        with ir.if_else() as ie:
            # REF branch: build CON, bind
            ir.const(TAG_CON); ir.const(val)
            ir.fn_call(FN_HEAP_PUSH); ir.set('gc_c')
            ir.local('gc_addr'); ir.const(TAG_REF); ir.fn_call(FN_HEAP_SET_TAG)
            ir.local('gc_addr'); ir.local('gc_c'); ir.fn_call(FN_HEAP_SET_VAL)
            _trail(ir, 'gc_addr')

            # else: must be CON with matching value
            ie.then_part()
            ir.local('gc_tag'); ir.const(TAG_CON); ir.eq()
            with ir.if_():
                ir.local('gc_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                ir.const(val); ir.ne()
                with ir.if_():
                    ir.const(1); ir.gset(G_FAIL)
            # neither REF nor CON: fail
            ir.local('gc_tag'); ir.const(TAG_REF); ir.ne()
            ir.local('gc_tag'); ir.const(TAG_CON); ir.ne()
            ir.and_()
            with ir.if_():
                ir.const(1); ir.gset(G_FAIL)

    # ==================================================================
    #  UNIFY instructions — subterm matching (read/write mode)
    # ==================================================================

    def _op_unify_variable(self, ir, args):
        vn = args[0]
        ir.new_local('uv')

        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            # read: reg[vn] = S
            _reg_set(ir, vn, lambda: ir.gget(G_S))
            # write: push self-ref REF
            ir.const(TAG_REF); ir.gget(G_H)
            ir.fn_call(FN_HEAP_PUSH); ir.set('uv')
            _reg_set(ir, vn, lambda: ir.local('uv'))

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_unify_value(self, ir, args):
        vn = args[0]

        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            # read: unify(reg[vn], S)
            _reg_get(ir, vn); ir.gget(G_S); ir.fn_call(FN_UNIFY)
            # write: copy cell onto heap
            _reg_get(ir, vn); ir.fn_call(FN_HEAP_GET_TAG)
            _reg_get(ir, vn); ir.fn_call(FN_HEAP_GET_VAL)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_unify_constant(self, ir, args):
        c = args[0]
        val = self.syms.encode_constant(c)
        ir.new_local('uc_addr'); ir.new_local('uc_c')

        ir.gget(G_MODE); ir.const(MODE_READ); ir.eq()
        with ir.if_else() as ie:
            # read mode
            ir.gget(G_S); ir.fn_call(FN_DEREF); ir.set('uc_addr')
            ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_TAG)
            ir.const(TAG_REF); ir.eq()
            with ir.if_else() as ie2:
                # REF: push CON, bind
                ir.const(TAG_CON); ir.const(val)
                ir.fn_call(FN_HEAP_PUSH); ir.set('uc_c')
                ir.local('uc_addr'); ir.const(TAG_REF)
                ir.fn_call(FN_HEAP_SET_TAG)
                ir.local('uc_addr'); ir.local('uc_c')
                ir.fn_call(FN_HEAP_SET_VAL)
                _trail(ir, 'uc_addr')
                # not REF: compare
                ie2.then_part()
                ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_TAG)
                ir.const(TAG_CON); ir.eq()
                with ir.if_():
                    ir.local('uc_addr'); ir.fn_call(FN_HEAP_GET_VAL)
                    ir.const(val); ir.ne()
                    with ir.if_():
                        ir.const(1); ir.gset(G_FAIL)

            # write mode: heap_push(TAG_CON, val)
            ir.const(TAG_CON); ir.const(val)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

        ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    def _op_unify_void(self, ir, args):
        n = args[0]
        for _ in range(n):
            ir.gget(G_MODE); ir.const(MODE_WRITE); ir.eq()
            with ir.if_():
                ir.const(TAG_REF); ir.gget(G_H)
                ir.fn_call(FN_HEAP_PUSH); ir.drop()
            ir.gget(G_S); ir.const(1); ir.add(); ir.gset(G_S)

    # ==================================================================
    #  SET instructions — structure building (write mode only)
    # ==================================================================

    def _op_set_variable(self, ir, args):
        vn = args[0]
        ir.new_local('sv')
        ir.const(TAG_REF); ir.gget(G_H)
        ir.fn_call(FN_HEAP_PUSH); ir.set('sv')
        _reg_set(ir, vn, lambda: ir.local('sv'))

    def _op_set_value(self, ir, args):
        vn = args[0]
        _reg_get(ir, vn); ir.fn_call(FN_HEAP_GET_TAG)
        _reg_get(ir, vn); ir.fn_call(FN_HEAP_GET_VAL)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()

    def _op_set_constant(self, ir, args):
        c = args[0]
        val = self.syms.encode_constant(c)
        ir.const(TAG_CON); ir.const(val)
        ir.fn_call(FN_HEAP_PUSH); ir.drop()

    def _op_set_void(self, ir, args):
        n = args[0]
        for _ in range(n):
            ir.const(TAG_REF); ir.gget(G_H)
            ir.fn_call(FN_HEAP_PUSH); ir.drop()

    # ==================================================================
    #  Control instructions
    # ==================================================================

    def _op_allocate(self, ir, args):
        n = args[0]
        # stack[ESTACK] = old_E
        ir.gget(G_STACK); ir.gget(G_ESTACK); ir.gget(G_E)
        ir._emit(array_set(T_STACK))
        # Yi slots = self-ref
        for i in range(n):
            ir.gget(G_STACK)
            ir.gget(G_ESTACK); ir.const(1 + i); ir.add()
            ir.gget(G_ESTACK); ir.const(1 + i); ir.add()
            ir._emit(array_set(T_STACK))
        ir.gget(G_ESTACK); ir.gset(G_E)
        ir.gget(G_ESTACK); ir.const(1 + n); ir.add(); ir.gset(G_ESTACK)

    def _op_deallocate(self, ir, args):
        ir.new_local('old_e')
        ir.gget(G_STACK); ir.gget(G_E); ir._emit(array_get(T_STACK))
        ir.set('old_e')
        ir.gget(G_E); ir.gset(G_ESTACK)
        ir.local('old_e'); ir.gset(G_E)

    def _op_call(self, ir, args):
        (name, arity) = args[0]
        key = f"{name}/{arity}"
        target = self.func_indices.get(key)
        if target is None:
            raise ValueError(f"unknown predicate: {key}")
        # Skip call if already failed; propagate failure after call.
        ir.gget(G_FAIL)
        with ir.if_():
            ir.ret()
        ir.fn_call(target)
        ir.gget(G_FAIL)
        with ir.if_():
            ir.ret()

    def _op_execute(self, ir, args):
        (name, arity) = args[0]
        key = f"{name}/{arity}"
        target = self.func_indices.get(key)
        if target is None:
            raise ValueError(f"unknown predicate: {key}")
        ir.return_call(target)

    def _op_proceed(self, ir, args):
        ir.ret()

    # ==================================================================
    #  Choice — Phase 5
    # ==================================================================

    def _op_try_me_else(self, ir, args):
        """Push a choice point; record the retry clause as the backtrack target.

        Choice point layout at G_ESTACK on G_STACK:
          [B+0] saved E
          [B+1] saved old G_B
          [B+2] saved TR
          [B+3] saved H
          [B+4] n  (arity)
          [B+5..B+4+n] saved X1..Xn
        """
        next_label, n = args
        next_func_idx = self.func_indices.get(next_label)
        if next_func_idx is None:
            raise ValueError(f"unknown label for try_me_else: {next_label}")
        self.declared_funcrefs.append(next_func_idx)

        ir.new_local('tme_b')
        ir.gget(G_ESTACK); ir.set('tme_b')

        # stack[b+0] = G_E
        ir.gget(G_STACK); ir.local('tme_b')
        ir.gget(G_E); ir._emit(array_set(T_STACK))
        # stack[b+1] = G_B  (old choice point)
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(1); ir.add()
        ir.gget(G_B); ir._emit(array_set(T_STACK))
        # stack[b+2] = G_TR
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(2); ir.add()
        ir.gget(G_TR); ir._emit(array_set(T_STACK))
        # stack[b+3] = G_H
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(3); ir.add()
        ir.gget(G_H); ir._emit(array_set(T_STACK))
        # stack[b+4] = n
        ir.gget(G_STACK); ir.local('tme_b'); ir.const(4); ir.add()
        ir.const(n); ir._emit(array_set(T_STACK))
        # stack[b+5..b+4+n] = X1..Xn
        for i in range(1, n + 1):
            ir.gget(G_STACK); ir.local('tme_b'); ir.const(4 + i); ir.add()
            emit_xget(ir, i)
            ir._emit(array_set(T_STACK))

        # BP_STACK[G_BP_TOP] = ref_func(next_func_idx)
        ir.gget(G_BP_STACK); ir.gget(G_BP_TOP)
        ir.ref_func(next_func_idx)
        ir._emit(array_set(T_CONT))
        ir.gget(G_BP_TOP); ir.const(1); ir.add(); ir.gset(G_BP_TOP)

        # G_B = b; G_ESTACK = b+5+n; G_HB = G_H
        ir.local('tme_b'); ir.gset(G_B)
        ir.local('tme_b'); ir.const(5 + n); ir.add(); ir.gset(G_ESTACK)
        ir.gget(G_H); ir.gset(G_HB)

    def _op_retry_me_else(self, ir, args):
        """Update the BP for the current choice point to the next clause.

        State has already been restored by the top-level backtrack driver
        before this clause function was called.
        """
        next_label, n = args
        next_func_idx = self.func_indices.get(next_label)
        if next_func_idx is None:
            raise ValueError(f"unknown label for retry_me_else: {next_label}")
        self.declared_funcrefs.append(next_func_idx)

        # BP_STACK[G_BP_TOP - 1] = ref_func(next_func_idx)
        ir.gget(G_BP_STACK)
        ir.gget(G_BP_TOP); ir.const(1); ir.sub()
        ir.ref_func(next_func_idx)
        ir._emit(array_set(T_CONT))

    def _op_trust_me(self, ir, args):
        """Pop the current choice point; no more alternatives.

        State has already been restored by the top-level backtrack driver.
        """
        ir.new_local('tm_old_b')
        # G_B = stack[G_B + 1]  (restore old choice point pointer)
        ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
        ir._emit(array_get(T_STACK))
        ir.set('tm_old_b')
        ir.local('tm_old_b'); ir.gset(G_B)
        # G_BP_TOP--
        ir.gget(G_BP_TOP); ir.const(1); ir.sub(); ir.gset(G_BP_TOP)

    def _op_neck_cut(self, ir, args):
        """Remove the current predicate's choice point, committing to this clause.

        The choice point at G_B stores old G_B in slot [B+1] (saved by
        try_me_else).  That value is the cut barrier for this predicate.
        Uses _cut_to so multi-level remnants are also cleaned up.
        """
        ir.new_local('nc_level')
        ir.gget(G_B); ir.const(-1); ir.ne()
        with ir.if_():
            # cut_level = old_B stored in the current choice point
            ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
            ir._emit(array_get(T_STACK))
            ir.set('nc_level')
            self._emit_cut_to(ir, 'nc_level')

    def _op_get_level(self, ir, args):
        """get_level Yn — save current G_B (cut barrier) into permanent var Yn."""
        yn = args[0]
        _reg_set(ir, yn, lambda: ir.gget(G_B))

    def _op_cut(self, ir, args):
        """cut Yn — remove all choice points created after the level saved in Yn."""
        yn = args[0]
        ir.new_local('cut_level')
        _reg_get(ir, yn); ir.set('cut_level')
        self._emit_cut_to(ir, 'cut_level')

    def _emit_cut_to(self, ir, level_local):
        """Emit a loop that pops choice points while G_B > level_local.

        Used by both neck_cut and cut.  level_local is the name of an i32
        local that holds the target B value (-1 = cut everything).
        """
        ir.new_local('_ct_old_b')
        with ir.while_loop() as loop:
            # stop when no choice point
            ir.gget(G_B); ir.const(-1); ir.eq(); loop.break_if()
            # stop when G_B <= level (already at or below cutoff)
            ir.gget(G_B); ir.local(level_local); ir.gt_s()
            ir.eqz(); loop.break_if()

            ir.gget(G_STACK); ir.gget(G_B); ir.const(1); ir.add()
            ir._emit(array_get(T_STACK))
            ir.set('_ct_old_b')
            ir.local('_ct_old_b'); ir.gset(G_B)
            ir.gget(G_BP_TOP); ir.const(1); ir.sub(); ir.gset(G_BP_TOP)
