"""WAM-to-WASM compiler: compiles WAM instructions to WASM 3.0 with GC.

Uses wir.py for structured control flow.

Runtime functions:
  0: init()
  1: heap_push(tag, value) -> addr
  2: heap_get_tag(addr) -> tag
  3: heap_get_val(addr) -> value
  4: heap_set_tag(addr, tag)
  5: heap_set_val(addr, value)
  6: deref(addr) -> addr
  7: bind(a1, a2)
  8: pdl_push(value)
  9: pdl_pop() -> value
 10: unify(a1, a2)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import (
    module, functype, comptype_array,
    subtype, reftype, byte,
    I32, global_entry, export_func, func_body,
    i32_const, local_get, local_set,
    global_get, global_set,
    array_new_default, array_get, array_set,
    ref_null, drop, call, return_,
    elem_declare, element_section,
    i32_add, i32_mul, i32_and, i32_or, i32_lt_s,
)
from wir import WIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_HEAP    = 0
T_TRAIL   = 1
T_STACK   = 2
T_PDL     = 3
T_XREG    = 4
FT_CLAUSE = 5   # functype [] -> [] — shared type for all compiled clause functions
T_CONT    = 6   # array of (ref null FT_CLAUSE) — continuation (ref) stack

TAG_REF = 0
TAG_CON = 1
TAG_STR = 2
TAG_LIS = 3
TAG_FUN = 4

G_H       = 0
G_E       = 1
G_B       = 2
G_HB      = 3
G_TR      = 4
G_S       = 5
G_MODE    = 6
G_FAIL    = 7
G_HALTED  = 8
G_ESTACK  = 9   # env stack top (next free slot on i32 stack)
G_HEAP    = 10
G_TRAIL   = 11
G_STACK   = 12
G_PDL_TOP  = 13
G_PDL      = 14
G_XREG     = 15
G_CONT_TOP  = 16
G_CONT      = 17
G_BP_TOP    = 18   # top of backtrack-point funcref stack
G_BP_STACK  = 19   # funcref array: next-clause references for choice points

HEAP_SIZE     = 65536
TRAIL_SIZE    = 16384
STACK_SIZE    = 16384
PDL_SIZE      = 4096
XREG_SIZE     = 256    # X1..X255; index 0 unused (WAM registers are 1-indexed)
CONT_SIZE     = 16384  # max call depth for continuation stack
BP_STACK_SIZE = 4096   # max simultaneous choice points

# Function indices (assigned in build_module)
FN_INIT               = 0
FN_HEAP_PUSH          = 1
FN_HEAP_GET_TAG       = 2
FN_HEAP_GET_VAL       = 3
FN_HEAP_SET_TAG       = 4
FN_HEAP_SET_VAL       = 5
FN_DEREF              = 6
FN_BIND               = 7
FN_PDL_PUSH           = 8
FN_PDL_POP            = 9
FN_UNIFY              = 10
FN_BACKTRACK_RESTORE  = 11  # restore state from current choice point
FN_TEST               = 12


# ---------------------------------------------------------------------------
# Types and globals
# ---------------------------------------------------------------------------

def runtime_types():
    return [
        subtype(comptype_array((I32, True))),                           # 0: T_HEAP
        subtype(comptype_array((I32, True))),                           # 1: T_TRAIL
        subtype(comptype_array((I32, True))),                           # 2: T_STACK
        subtype(comptype_array((I32, True))),                           # 3: T_PDL
        subtype(comptype_array((I32, True))),                           # 4: T_XREG
        functype([], []),                                               # 5: FT_CLAUSE
        subtype(byte(0x5E) + reftype(True, FT_CLAUSE) + byte(0x01)),      # 6: T_CONT (array of (ref null FT_CLAUSE), mutable)
    ]


def runtime_globals():
    return [
        global_entry(I32, True, [i32_const(0)]) for _ in range(10)
    ] + [
        global_entry(reftype(True, T_HEAP),  True, [ref_null(T_HEAP)]),   # G_HEAP
        global_entry(reftype(True, T_TRAIL), True, [ref_null(T_TRAIL)]),  # G_TRAIL
        global_entry(reftype(True, T_STACK), True, [ref_null(T_STACK)]),  # G_STACK
        global_entry(I32, True, [i32_const(0)]),                          # G_PDL_TOP
        global_entry(reftype(True, T_PDL),   True, [ref_null(T_PDL)]),    # G_PDL
        global_entry(reftype(True, T_XREG),  True, [ref_null(T_XREG)]),  # G_XREG
        global_entry(I32, True, [i32_const(0)]),                          # G_CONT_TOP
        global_entry(reftype(True, T_CONT),  True, [ref_null(T_CONT)]),   # G_CONT
        global_entry(I32, True, [i32_const(0)]),                          # G_BP_TOP
        global_entry(reftype(True, T_CONT),  True, [ref_null(T_CONT)]),   # G_BP_STACK
    ]


# ---------------------------------------------------------------------------
# X register access helpers
#
# X registers are 1-indexed heap addresses (index 0 is unused).
# Layout: G_XREG[i] = heap address held in Xi, for i in 1..XREG_SIZE-1.
#
# These are Python helpers that emit inline array access — not WASM functions,
# so there is no call overhead. Usage:
#
#   emit_xget(ir, 1)          # push X1's value (a heap address) onto the stack
#   emit_xset(ir, 1, lambda: ir.local('addr'))  # X1 = addr
# ---------------------------------------------------------------------------

def emit_xget(ir, n):
    """Emit: push xreg[n] (a heap address) onto the stack."""
    ir.gget(G_XREG); ir.const(n); ir._emit(array_get(T_XREG))

def emit_xset(ir, n, emit_value):
    """Emit: xreg[n] = <value>. emit_value() must push the value onto the stack."""
    ir.gget(G_XREG); ir.const(n); emit_value(); ir._emit(array_set(T_XREG))


# ---------------------------------------------------------------------------
# Continuation stack helpers
#
# The continuation stack is a separate GC array of (ref null FT_CLAUSE).
# It holds the return continuation for each active WAM call.
#
# i32 stack  [E+0: old_E | E+1: n_perm | E+2: Y1 | ...]  (environment data)
# cont stack [E+0: return_cp            (ref null FT_CLAUSE) ]
#
# Both stacks share the same top-of-stack pointer (G_E / the environment
# register), so they grow and shrink together.  G_CONT_TOP is a separate
# pointer used only for call/proceed; environment allocation uses G_E.
#
# Usage:
#   emit_cont_push(ir, lambda: ir.ref_func(funcidx))  # push a real continuation
#   emit_cont_push(ir, lambda: ir.ref_null(FT_CLAUSE)) # push null (query entry)
#   emit_cont_pop(ir)                                  # pop; leaves ref on stack
#   ir.call_ref(FT_CLAUSE)                             # call whatever is on stack
# ---------------------------------------------------------------------------

def emit_cont_push(ir, emit_value):
    """Push a (ref null FT_CLAUSE) onto the continuation stack.

    emit_value() must push the funcref value onto the WASM stack.
    """
    ir.gget(G_CONT); ir.gget(G_CONT_TOP); emit_value()
    ir._emit(array_set(T_CONT))
    ir.gget(G_CONT_TOP); ir.const(1); ir.add(); ir.gset(G_CONT_TOP)

def emit_cont_pop(ir):
    """Pop the top funcref from the continuation stack onto the WASM stack."""
    ir.gget(G_CONT_TOP); ir.const(1); ir.sub(); ir.gset(G_CONT_TOP)
    ir.gget(G_CONT); ir.gget(G_CONT_TOP); ir._emit(array_get(T_CONT))


# ---------------------------------------------------------------------------
# Runtime functions using WIR
# ---------------------------------------------------------------------------

def build_init():
    """init() — create heap/trail/stack/pdl/xreg/cont/bp arrays; reset registers."""
    ir = WIR([])
    ir.const(HEAP_SIZE);     ir._emit(array_new_default(T_HEAP));  ir.gset(G_HEAP)
    ir.const(TRAIL_SIZE);    ir._emit(array_new_default(T_TRAIL)); ir.gset(G_TRAIL)
    ir.const(STACK_SIZE);    ir._emit(array_new_default(T_STACK)); ir.gset(G_STACK)
    ir.const(PDL_SIZE);      ir._emit(array_new_default(T_PDL));   ir.gset(G_PDL)
    ir.const(XREG_SIZE);     ir._emit(array_new_default(T_XREG));  ir.gset(G_XREG)
    ir.const(CONT_SIZE);     ir._emit(array_new_default(T_CONT));  ir.gset(G_CONT)
    ir.const(BP_STACK_SIZE); ir._emit(array_new_default(T_CONT));  ir.gset(G_BP_STACK)
    # G_B = -1 means "no choice point"
    ir.const(-1); ir.gset(G_B)
    ir.const(0);  ir.gset(G_BP_TOP)
    ir.const(0);  ir.gset(G_H)
    ir.const(0);  ir.gset(G_E)
    ir.const(0);  ir.gset(G_ESTACK)
    ir.const(0);  ir.gset(G_TR)
    ir.const(0);  ir.gset(G_FAIL)
    return functype([], []), ir.encode()


def build_heap_push():
    """heap_push(tag: i32, value: i32) -> addr. Cell = 2 i32 slots."""
    ir = WIR(['tag', 'val'], I32)
    ir.new_local('old_h')

    # old_h = H
    ir.gget(G_H); ir.set('old_h')

    # heap[2*old_h] = tag
    ir.gget(G_HEAP); ir.local('old_h'); ir.const(2); ir.mul()
    ir.local('tag'); ir._emit(array_set(T_HEAP))

    # heap[2*old_h + 1] = val
    ir.gget(G_HEAP); ir.local('old_h'); ir.const(2); ir.mul(); ir.const(1); ir.add()
    ir.local('val'); ir._emit(array_set(T_HEAP))

    # H = old_h + 1
    ir.local('old_h'); ir.const(1); ir.add(); ir.gset(G_H)

    ir.local('old_h')
    return functype([I32, I32], [I32]), ir.encode()


def build_heap_get_tag():
    """heap_get_tag(addr: i32) -> i32."""
    ir = WIR(['addr'], I32)
    ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul()
    ir._emit(array_get(T_HEAP))
    return functype([I32], [I32]), ir.encode()


def build_heap_get_val():
    """heap_get_val(addr: i32) -> i32."""
    ir = WIR(['addr'], I32)
    ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul(); ir.const(1); ir.add()
    ir._emit(array_get(T_HEAP))
    return functype([I32], [I32]), ir.encode()


def build_heap_set_tag():
    """heap_set_tag(addr: i32, tag: i32)."""
    ir = WIR(['addr', 'tag'])
    ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul()
    ir.local('tag'); ir._emit(array_set(T_HEAP))
    return functype([I32, I32], []), ir.encode()


def build_heap_set_val():
    """heap_set_val(addr: i32, val: i32)."""
    ir = WIR(['addr', 'val'])
    ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul(); ir.const(1); ir.add()
    ir.local('val'); ir._emit(array_set(T_HEAP))
    return functype([I32, I32], []), ir.encode()


def build_deref():
    """deref(addr: i32) -> i32. Chase REF chain until non-REF or self-ref.

    while True:
        tag = heap_get_tag(addr)
        if tag != REF: return addr
        val = heap_get_val(addr)
        if val == addr: return addr
        addr = val
    """
    ir = WIR(['addr'], I32)
    ir.new_local('tag')
    ir.new_local('val')

    with ir.while_loop():
        # tag = heap_get_tag(addr)
        ir.local('addr'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('tag')

        # if tag != REF: return addr
        ir.local('tag'); ir.const(TAG_REF); ir.ne()
        with ir.if_():
            ir.local('addr'); ir.ret()

        # val = heap_get_val(addr)
        ir.local('addr'); ir.fn_call(FN_HEAP_GET_VAL); ir.set('val')

        # if val == addr: return addr
        ir.local('val'); ir.local('addr'); ir.eq()
        with ir.if_():
            ir.local('addr'); ir.ret()

        # addr = val
        ir.local('val'); ir.set('addr')

    # Unreachable but needed for validation
    ir.local('addr'); ir.ret()

    return functype([I32], [I32]), ir.encode()


def _trail(ir, addr_local):
    """Emit: if addr < HB, push addr onto trail and increment TR."""
    ir.local(addr_local); ir.gget(G_HB); ir.lt_s()
    with ir.if_():
        ir.gget(G_TRAIL); ir.gget(G_TR); ir.local(addr_local)
        ir._emit(array_set(T_TRAIL))
        ir.gget(G_TR); ir.const(1); ir.add(); ir.gset(G_TR)


def build_bind():
    """bind(a1: i32, a2: i32). Bind two heap cells, trail the modification.

    if cell[a1].tag == REF and (cell[a2].tag != REF or a2 < a1):
        cell[a1] = cell[a2]   # bind a1 toward a2
        trail(a1)
    else:
        cell[a2] = cell[a1]   # bind a2 toward a1
        trail(a2)
    """
    ir = WIR(['a1', 'a2'])
    ir.new_local('tag1')
    ir.new_local('val1')
    ir.new_local('tag2')
    ir.new_local('val2')

    # Read both cells
    ir.local('a1'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('tag1')
    ir.local('a1'); ir.fn_call(FN_HEAP_GET_VAL); ir.set('val1')
    ir.local('a2'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('tag2')
    ir.local('a2'); ir.fn_call(FN_HEAP_GET_VAL); ir.set('val2')

    # Condition: tag1==REF AND (tag2!=REF OR a2<a1)
    ir.local('tag1'); ir.const(TAG_REF); ir.eq()     # A
    ir.local('tag2'); ir.const(TAG_REF); ir.ne()      # B
    ir.local('a2'); ir.local('a1'); ir.lt_s()         # C
    ir.or_()                                          # B | C
    ir.and_()                                         # A & (B|C)

    with ir.if_else() as ie:
        # bind a1 → c2
        ir.local('a1'); ir.local('tag2'); ir.fn_call(FN_HEAP_SET_TAG)
        ir.local('a1'); ir.local('val2'); ir.fn_call(FN_HEAP_SET_VAL)
        _trail(ir, 'a1')
        # else: bind a2 → c1
        ie.then_part()
        ir.local('a2'); ir.local('tag1'); ir.fn_call(FN_HEAP_SET_TAG)
        ir.local('a2'); ir.local('val1'); ir.fn_call(FN_HEAP_SET_VAL)
        _trail(ir, 'a2')

    return functype([I32, I32], []), ir.encode()


def build_pdl_push():
    """pdl_push(value: i32). Push a value onto the PDL stack."""
    ir = WIR(['val'])
    ir.gget(G_PDL); ir.gget(G_PDL_TOP); ir.local('val')
    ir._emit(array_set(T_PDL))
    ir.gget(G_PDL_TOP); ir.const(1); ir.add(); ir.gset(G_PDL_TOP)
    return functype([I32], []), ir.encode()


def build_pdl_pop():
    """pdl_pop() -> i32. Pop a value from the PDL stack."""
    ir = WIR([], I32)
    ir.gget(G_PDL_TOP); ir.const(1); ir.sub(); ir.gset(G_PDL_TOP)
    ir.gget(G_PDL); ir.gget(G_PDL_TOP); ir._emit(array_get(T_PDL))
    return functype([], [I32]), ir.encode()


def build_unify():
    """unify(a1: i32, a2: i32). Full unification with PDL.

    Sets G_FAIL to 1 if unification fails.

    Mirrors the reference WAM unify algorithm:
      - Push a1, a2 onto PDL
      - While PDL not empty and not failed:
          pop and deref both addresses
          if same address: skip
          dispatch on tag combination:
            REF:     bind
            CON+CON: compare values
            LIS+LIS: push car/cdr pairs
            STR+STR: compare functors, push arg pairs
            else:    fail
    """
    ir = WIR(['a1', 'a2'])
    ir.new_local('d1')
    ir.new_local('d2')
    ir.new_local('tag1')
    ir.new_local('tag2')
    ir.new_local('val1')
    ir.new_local('val2')
    ir.new_local('i')
    ir.new_local('arity')

    # Reset PDL and fail flag
    ir.const(0); ir.gset(G_PDL_TOP)
    ir.const(0); ir.gset(G_FAIL)

    # Push initial pair
    ir.local('a1'); ir.fn_call(FN_PDL_PUSH)
    ir.local('a2'); ir.fn_call(FN_PDL_PUSH)

    # while PDL_TOP > 0 and not FAIL:
    with ir.while_loop() as loop:
        ir.gget(G_PDL_TOP); ir.eqz();  loop.break_if()
        ir.gget(G_FAIL);               loop.break_if()

        # d1 = deref(pop()); d2 = deref(pop())
        ir.fn_call(FN_PDL_POP); ir.fn_call(FN_DEREF); ir.set('d1')
        ir.fn_call(FN_PDL_POP); ir.fn_call(FN_DEREF); ir.set('d2')

        # if d1 == d2: skip (continue to next iteration)
        ir.local('d1'); ir.local('d2'); ir.ne()
        with ir.if_():
            # Read both cells
            ir.local('d1'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('tag1')
            ir.local('d1'); ir.fn_call(FN_HEAP_GET_VAL); ir.set('val1')
            ir.local('d2'); ir.fn_call(FN_HEAP_GET_TAG); ir.set('tag2')
            ir.local('d2'); ir.fn_call(FN_HEAP_GET_VAL); ir.set('val2')

            with ir.block() as dispatch:

                # case: tag1 == REF → bind(d1, d2)
                ir.local('tag1'); ir.const(TAG_REF); ir.eq()
                with ir.if_():
                    ir.local('d1'); ir.local('d2'); ir.fn_call(FN_BIND)
                    dispatch.break_()

                # case: tag2 == REF → bind(d2, d1)
                ir.local('tag2'); ir.const(TAG_REF); ir.eq()
                with ir.if_():
                    ir.local('d2'); ir.local('d1'); ir.fn_call(FN_BIND)
                    dispatch.break_()

                # case: CON + CON → compare values
                ir.local('tag1'); ir.const(TAG_CON); ir.eq()
                ir.local('tag2'); ir.const(TAG_CON); ir.eq()
                ir.and_()
                with ir.if_():
                    ir.local('val1'); ir.local('val2'); ir.ne()
                    with ir.if_():
                        ir.const(1); ir.gset(G_FAIL)
                    dispatch.break_()

                # case: LIS + LIS → push car/cdr pairs
                ir.local('tag1'); ir.const(TAG_LIS); ir.eq()
                ir.local('tag2'); ir.const(TAG_LIS); ir.eq()
                ir.and_()
                with ir.if_():
                    ir.local('val1'); ir.fn_call(FN_PDL_PUSH)
                    ir.local('val2'); ir.fn_call(FN_PDL_PUSH)
                    ir.local('val1'); ir.const(1); ir.add()
                    ir.fn_call(FN_PDL_PUSH)
                    ir.local('val2'); ir.const(1); ir.add()
                    ir.fn_call(FN_PDL_PUSH)
                    dispatch.break_()

                # case: STR + STR → compare functors, push arg pairs
                ir.local('tag1'); ir.const(TAG_STR); ir.eq()
                ir.local('tag2'); ir.const(TAG_STR); ir.eq()
                ir.and_()
                with ir.if_():
                    # Read packed functor values (FUN cells at val1/val2)
                    # Packed = functor_id << 8 | arity
                    ir.local('val1'); ir.fn_call(FN_HEAP_GET_VAL)
                    ir.set('i')  # temporarily holds fc1_packed
                    ir.local('val2'); ir.fn_call(FN_HEAP_GET_VAL)
                    ir.local('i'); ir.ne()
                    with ir.if_():
                        # Functors differ → fail
                        ir.const(1); ir.gset(G_FAIL)
                        dispatch.break_()

                    # Extract arity from packed value
                    ir.local('i'); ir.const(0xFF); ir.and_()
                    ir.set('arity')

                    # for i = 1; i <= arity; i++:
                    ir.const(1); ir.set('i')
                    with ir.while_loop() as arg_loop:
                        ir.local('i'); ir.local('arity'); ir.gt_s()
                        arg_loop.break_if()
                        # push val1+i and val2+i
                        ir.local('val1'); ir.local('i'); ir.add()
                        ir.fn_call(FN_PDL_PUSH)
                        ir.local('val2'); ir.local('i'); ir.add()
                        ir.fn_call(FN_PDL_PUSH)
                        ir.local('i'); ir.const(1); ir.add(); ir.set('i')

                    dispatch.break_()

                # default: type mismatch → fail
                ir.const(1); ir.gset(G_FAIL)

    return functype([I32, I32], []), ir.encode()


def build_backtrack_restore():
    """backtrack_restore() — restore machine state from the current choice point.

    Choice point layout at G_B on G_STACK:
      [B+0] saved E
      [B+1] saved old G_B (-1 if none)
      [B+2] saved TR
      [B+3] saved H  (= HB at choice-point creation time)
      [B+4] n        (arity: number of saved X registers)
      [B+5..B+4+n]   saved X1..Xn

    Steps:
      1. Restore G_E
      2. Unwind trail from current G_TR down to saved TR, resetting each cell
      3. Reset G_H = G_HB = saved H; reset G_TR = saved TR
      4. Restore X1..Xn
      5. Clear G_FAIL

    Does NOT update G_B or G_BP_TOP (those are updated by trust_me at the
    start of the last retry clause).
    """
    ir = WIR([])
    ir.new_local('b')
    ir.new_local('n')
    ir.new_local('i')
    ir.new_local('addr')
    ir.new_local('saved_tr')
    ir.new_local('saved_h')

    # b = G_B
    ir.gget(G_B); ir.set('b')

    # Restore E
    ir.gget(G_STACK); ir.local('b'); ir._emit(array_get(T_STACK))
    ir.gset(G_E)

    # saved_tr = stack[b+2]
    ir.gget(G_STACK); ir.local('b'); ir.const(2); ir.add()
    ir._emit(array_get(T_STACK))
    ir.set('saved_tr')

    # saved_h = stack[b+3]
    ir.gget(G_STACK); ir.local('b'); ir.const(3); ir.add()
    ir._emit(array_get(T_STACK))
    ir.set('saved_h')

    # Unwind trail: while G_TR > saved_tr
    with ir.while_loop() as loop:
        ir.gget(G_TR); ir.local('saved_tr'); ir.gt_s()
        ir.eqz(); loop.break_if()      # break when G_TR <= saved_tr

        ir.gget(G_TR); ir.const(1); ir.sub(); ir.gset(G_TR)
        ir.gget(G_TRAIL); ir.gget(G_TR); ir._emit(array_get(T_TRAIL))
        ir.set('addr')
        # heap[addr] = (REF, addr)   — self-referential = unbound
        ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul()
        ir.const(TAG_REF); ir._emit(array_set(T_HEAP))
        ir.gget(G_HEAP); ir.local('addr'); ir.const(2); ir.mul(); ir.const(1); ir.add()
        ir.local('addr'); ir._emit(array_set(T_HEAP))

    # Reset H, HB, TR
    ir.local('saved_h'); ir.gset(G_H)
    ir.local('saved_h'); ir.gset(G_HB)
    ir.local('saved_tr'); ir.gset(G_TR)

    # n = stack[b+4]
    ir.gget(G_STACK); ir.local('b'); ir.const(4); ir.add()
    ir._emit(array_get(T_STACK))
    ir.set('n')

    # Restore X1..Xn
    ir.const(1); ir.set('i')
    with ir.while_loop() as loop2:
        ir.local('i'); ir.local('n'); ir.gt_s()
        loop2.break_if()   # break when i > n

        # xreg[i] = stack[b + 4 + i]
        ir.gget(G_XREG); ir.local('i')
        ir.gget(G_STACK)
        ir.local('b'); ir.const(4); ir.add(); ir.local('i'); ir.add()
        ir._emit(array_get(T_STACK))
        ir._emit(array_set(T_XREG))

        ir.local('i'); ir.const(1); ir.add(); ir.set('i')

    # Clear G_FAIL
    ir.const(0); ir.gset(G_FAIL)

    return functype([], []), ir.encode()


# ---------------------------------------------------------------------------
# Build module
# ---------------------------------------------------------------------------

def build_module():
    types = runtime_types()

    # Build all functions
    fns = [
        build_init,               # 0
        build_heap_push,          # 1
        build_heap_get_tag,       # 2
        build_heap_get_val,       # 3
        build_heap_set_tag,       # 4
        build_heap_set_val,       # 5
        build_deref,              # 6
        build_bind,               # 7
        build_pdl_push,           # 8
        build_pdl_pop,            # 9
        build_unify,              # 10
        build_backtrack_restore,  # 11
    ]

    func_types = []
    func_codes = []
    func_type_indices = []

    for builder in fns:
        ft, code = builder()
        ti = len(types)
        types.append(ft)
        func_type_indices.append(ti)
        func_codes.append(code)

    # Test function: unify f(X) with f(42), return X (should be 42)
    test_type = functype([], [I32])
    test_ti = len(types)
    types.append(test_type)

    ir = WIR([], I32)
    ir.fn_call(FN_INIT)

    # Build f(X) on heap:
    #   addr 0: FUN(f/1) — tag=FUN, val=packed(id=1, arity=1) = 0x101
    #   addr 1: REF(1)   — unbound variable X
    #   addr 2: STR(0)   — structure pointing to functor at 0
    ir.const(TAG_FUN); ir.const(0x101); ir.fn_call(FN_HEAP_PUSH); ir.drop()
    ir.const(TAG_REF); ir.const(1);     ir.fn_call(FN_HEAP_PUSH); ir.drop()
    ir.const(TAG_STR); ir.const(0);     ir.fn_call(FN_HEAP_PUSH); ir.drop()

    # Build f(42) on heap:
    #   addr 3: FUN(f/1) — same functor
    #   addr 4: CON(42)  — constant 42
    #   addr 5: STR(3)   — structure pointing to functor at 3
    ir.const(TAG_FUN); ir.const(0x101); ir.fn_call(FN_HEAP_PUSH); ir.drop()
    ir.const(TAG_CON); ir.const(42);    ir.fn_call(FN_HEAP_PUSH); ir.drop()
    ir.const(TAG_STR); ir.const(3);     ir.fn_call(FN_HEAP_PUSH); ir.drop()

    # unify(2, 5): should bind X (addr 1) to CON(42)
    ir.const(2); ir.const(5); ir.fn_call(FN_UNIFY)

    # Return heap_get_val(deref(1)) — should be 42
    ir.const(1); ir.fn_call(FN_DEREF); ir.fn_call(FN_HEAP_GET_VAL)

    func_type_indices.append(test_ti)
    func_codes.append(ir.encode())

    # Test 2: cont stack round-trip.
    # A clause-typed helper that writes 99 into X1.
    # The test pushes its funcref onto the cont stack, pops it, calls it,
    # then reads X1 back — expecting 99.
    helper_clause_type = functype([], [])
    helper_ti = len(types)
    types.append(helper_clause_type)
    helper_ir = WIR([])
    emit_xset(helper_ir, 1, lambda: helper_ir.const(99))
    func_type_indices.append(helper_ti)
    helper_funcidx = len(func_type_indices) - 1  # index of this helper
    func_codes.append(helper_ir.encode())

    cont_test_type = functype([], [I32])
    cont_test_ti = len(types)
    types.append(cont_test_type)

    cir = WIR([], I32)
    cir.fn_call(FN_INIT)
    # push ref to helper onto cont stack, pop it, call it
    emit_cont_push(cir, lambda: cir.ref_func(helper_funcidx))
    emit_cont_pop(cir)
    cir.call_ref(FT_CLAUSE)
    # read X1 — should be 99
    emit_xget(cir, 1)

    func_type_indices.append(cont_test_ti)
    FN_CONT_TEST = len(func_type_indices) - 1
    func_codes.append(cir.encode())

    # Declarative element segment: all functions referenced via ref.func must
    # be listed here, or the validator rejects the module.
    declared_funcrefs = [helper_funcidx]

    return module(
        types=types,
        funcs=func_type_indices,
        globals_=runtime_globals(),
        codes=func_codes,
        exports=[
            export_func("init",          FN_INIT),
            export_func("heap_push",     FN_HEAP_PUSH),
            export_func("heap_get_tag",  FN_HEAP_GET_TAG),
            export_func("heap_get_val",  FN_HEAP_GET_VAL),
            export_func("heap_set_tag",  FN_HEAP_SET_TAG),
            export_func("heap_set_val",  FN_HEAP_SET_VAL),
            export_func("deref",         FN_DEREF),
            export_func("bind",          FN_BIND),
            export_func("unify",         FN_UNIFY),
            export_func("test",          FN_TEST),
            export_func("cont_test",     FN_CONT_TEST),
        ],
        elements=[elem_declare(declared_funcrefs)],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_wasm(wasm_bytes, label):
    import subprocess, tempfile, os
    path = tempfile.mktemp(suffix=".wasm")
    with open(path, 'wb') as f:
        f.write(wasm_bytes)
    try:
        r = subprocess.run(
            ["wasmtime", "compile", "-W", "all-proposals=y", path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"FAIL {label}:\n{r.stderr}")
            return False
        print(f"  ok  {label}")
        return True
    finally:
        os.unlink(path)


def run_wasm(wasm_bytes, func_name, *args):
    import subprocess, tempfile, os
    path = tempfile.mktemp(suffix=".wasm")
    with open(path, 'wb') as f:
        f.write(wasm_bytes)
    try:
        cmd = [
            "wasmtime",
            "-W", "all-proposals=y",
            "--invoke", func_name, path,
        ] + [str(a) for a in args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"RUN ERROR:\n{r.stderr}")
            return None
        return int(r.stdout.strip())
    finally:
        os.unlink(path)


if __name__ == "__main__":
    wasm = build_module()
    with open("stage2.wasm", "wb") as f:
        f.write(wasm)
    print(f"Module: {len(wasm)} bytes")
    validate_wasm(wasm, "runtime + unify + cont stack")
    result = run_wasm(wasm, "test")
    print(f"  test()      = {result}  (unify f(X) with f(42), expected 42)")
    result2 = run_wasm(wasm, "cont_test")
    print(f"  cont_test() = {result2}  (push/pop/call_ref clause fn, expected 99)")
