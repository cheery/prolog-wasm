"""WASM IR: tiny imperative builder for WASM function bodies.

Lets you write WASM functions like small Python programs:

    ir = WIR(['addr'], I32)
    ir.new_local('tag')

    with ir.while_loop():
        ir.local('addr'); ir.call(FN_HEAP_GET_TAG); ir.set('tag')
        ir.local('tag'); ir.const(TAG_REF); ir.ne()
        with ir.if_():
            ir.local('addr'); ir.ret()
        ir.local('addr'); ir.call(FN_HEAP_GET_VAL); ir.set('addr')

    ir.local('addr'); ir.ret()

Compiles to raw WASM instruction bytes via .encode().
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import (
    byte, u32, blocktype, I32,
    i32_const, local_get, local_set, local_tee,
    global_get, global_set,
    call as _call, drop as _drop, return_ as _return,
    return_call as _return_call,
    ref_func as _ref_func, ref_null as _ref_null,
    call_ref as _call_ref, return_call_ref as _return_call_ref,
    i32_add, i32_sub, i32_mul,
    i32_div_s as _i32_div_s, i32_rem_s as _i32_rem_s,
    i32_and as _i32_and, i32_or as _i32_or,
    i32_eq, i32_ne, i32_eqz,
    i32_lt_s, i32_lt_u, i32_le_s as _i32_le_s,
    i32_gt_s, i32_gt_u as _i32_gt_u,
    i32_ge_s, i32_ge_u as _i32_ge_u,
    unreachable as _unreachable,
    func_body,
)


class WIR:
    """WASM function body builder with named locals and structured control flow."""

    def __init__(self, params, result=None, results=None):
        """
        params:  list of names for i32 parameters
        result:  None for void, I32 for single i32 result (legacy)
        results: list of value types for multi-value return (overrides result)
        """
        self._instrs = []
        self._locals = {}
        self._new_locals = []     # [(count, I32), ...] for func_body
        self._next_local = len(params)
        self._depth = 0           # control-flow nesting depth for label calc

        if results is not None:
            self._results = results
        elif result is not None:
            self._results = [result]
        else:
            self._results = []

        for i, name in enumerate(params):
            self._locals[name] = i

    def new_local(self, name):
        """Declare a new i32 local. Returns its index."""
        idx = self._next_local
        self._next_local += 1
        self._locals[name] = idx
        self._new_locals.append((1, I32))
        return idx

    def new_local_ref(self, name, type_bytes):
        """Declare a new ref-typed local. Returns its index."""
        idx = self._next_local
        self._next_local += 1
        self._locals[name] = idx
        self._new_locals.append((1, type_bytes))
        return idx

    # -- emit raw bytes --
    def _emit(self, b: bytes):
        self._instrs.append(b)

    # -- constants --
    def const(self, n):
        self._emit(i32_const(n))

    # -- locals --
    def local(self, name):
        """Push named local's value."""
        self._emit(local_get(self._locals[name]))

    def set(self, name):
        """Pop stack into named local."""
        self._emit(local_set(self._locals[name]))

    def tee(self, name):
        """Pop into named local, keep on stack."""
        self._emit(local_tee(self._locals[name]))

    # -- globals --
    def gget(self, idx):
        self._emit(global_get(idx))

    def gset(self, idx):
        self._emit(global_set(idx))

    # -- calls --
    def fn_call(self, idx):
        self._emit(_call(idx))

    # -- stack --
    def drop(self):
        self._emit(_drop())

    def ret(self):
        self._emit(_return())

    def return_call(self, funcidx):
        """Tail-call a function directly (no stack frame)."""
        self._emit(_return_call(funcidx))

    # -- funcref instructions --
    def ref_func(self, funcidx):
        """Push a non-null reference to a specific function."""
        self._emit(_ref_func(funcidx))

    def ref_null(self, heaptype):
        """Push a null reference of the given heap type (a type index)."""
        self._emit(_ref_null(heaptype))

    def call_ref(self, typeidx):
        """Call through a (ref null $t) on the stack. Traps if null."""
        self._emit(_call_ref(typeidx))

    def return_call_ref(self, typeidx):
        """Tail-call through a (ref null $t) on the stack. Traps if null."""
        self._emit(_return_call_ref(typeidx))

    def unreachable(self):
        self._emit(_unreachable())

    # -- i32 arithmetic --
    def add(self):   self._emit(i32_add())
    def sub(self):   self._emit(i32_sub())
    def mul(self):   self._emit(i32_mul())
    def div_s(self): self._emit(_i32_div_s())
    def rem_s(self): self._emit(_i32_rem_s())

    # -- i32 bitwise --
    def and_(self): self._emit(_i32_and())
    def or_(self):  self._emit(_i32_or())

    # -- i32 comparison --
    def eq(self):   self._emit(i32_eq())
    def ne(self):   self._emit(i32_ne())
    def eqz(self):  self._emit(i32_eqz())
    def lt_s(self): self._emit(i32_lt_s())
    def lt_u(self): self._emit(i32_lt_u())
    def le_s(self): self._emit(_i32_le_s())
    def gt_s(self): self._emit(i32_gt_s())
    def gt_u(self): self._emit(_i32_gt_u())
    def ge_s(self): self._emit(i32_ge_s())
    def ge_u(self): self._emit(_i32_ge_u())

    # -- control flow --
    def block(self):
        """Context manager. Creates a block scope; use b.break_() to exit early.

        Enables the early-exit dispatch pattern (flat if-chains instead
        of deeply nested if-else):

            with ir.block() as b:
                ir.local('tag'); ir.const(TAG_REF); ir.eq()
                with ir.if_():
                    ...handle REF...
                    b.break_()
                ir.local('tag'); ir.const(TAG_CON); ir.eq()
                with ir.if_():
                    ...handle CON...
                    b.break_()
                ...default / fallthrough...
        """
        return _Block(self)

    def while_loop(self):
        """Context manager. Body repeats; use w.break_if() to exit.

        Compiles to:
            block void
              loop void
                <body>
                br 0          ;; implicit continue
              end
            end
        """
        return _While(self)

    def if_(self):
        """Context manager. Pops condition (i32); enters body if nonzero."""
        return _If(self)

    def if_else(self, bt=None):
        """Context manager. Pops condition; yields (then_ir, else_ir).

        bt: block type — None for void, I32 for single i32,
            or a type index for multi-value results.
        """
        return _IfElse(self, bt)

    # -- encode --

    def encode(self):
        """Return func_body bytes ready for the code section."""
        return func_body(self._new_locals, self._instrs)


class _Block:
    """block { body } end — early-exit scope."""

    def __init__(self, wir):
        self.wir = wir

    def __enter__(self):
        self.wir._emit(byte(0x02) + blocktype(None))    # block void
        self.wir._depth += 1
        self._depth = self.wir._depth
        return self

    def __exit__(self, *a):
        self.wir._depth -= 1
        self.wir._emit(byte(0x0B))                      # end

    def break_(self):
        """Unconditionally exit this block."""
        self.wir._emit(byte(0x0C) + u32(self.wir._depth - self._depth))

    def break_if(self):
        """Pop i32; if nonzero, exit this block."""
        self.wir._emit(byte(0x0D) + u32(self.wir._depth - self._depth))


class _While:
    """block { loop { body; br 0 } }"""

    def __init__(self, wir):
        self.wir = wir

    def __enter__(self):
        self.wir._emit(byte(0x02) + blocktype(None))    # block void
        self.wir._depth += 1
        self._block_depth = self.wir._depth
        self.wir._emit(byte(0x03) + blocktype(None))    # loop void
        self.wir._depth += 1
        self._loop_depth = self.wir._depth
        return self

    def __exit__(self, *a):
        # br 0 = continue (back to loop top)
        self.wir._emit(byte(0x0C) + u32(self.wir._depth - self._loop_depth))
        self.wir._depth -= 1
        self.wir._emit(byte(0x0B))                      # end loop
        self.wir._depth -= 1
        self.wir._emit(byte(0x0B))                      # end block

    def break_if(self):
        """Pop i32; if nonzero, exit the while loop."""
        self.wir._emit(byte(0x0D) + u32(self.wir._depth - self._block_depth))

    def break_(self):
        """Unconditionally exit the while loop."""
        self.wir._emit(byte(0x0C) + u32(self.wir._depth - self._block_depth))


class _If:
    """if <cond> <body> end"""

    def __init__(self, wir):
        self.wir = wir

    def __enter__(self):
        self.wir._emit(byte(0x04) + blocktype(None))    # if void
        self.wir._depth += 1
        return self

    def __exit__(self, *a):
        self.wir._depth -= 1
        self.wir._emit(byte(0x0B))                      # end


class _IfElse:
    """if <cond> <then> else <else> end"""

    def __init__(self, wir, bt=None):
        self.wir = wir
        self._bt = bt

    def __enter__(self):
        self.wir._emit(byte(0x04) + blocktype(self._bt))
        self.wir._depth += 1
        return self

    def then_part(self):
        """Call before emitting else."""
        self.wir._emit(byte(0x05))                      # else

    def __exit__(self, *a):
        self.wir._depth -= 1
        self.wir._emit(byte(0x0B))                      # end
