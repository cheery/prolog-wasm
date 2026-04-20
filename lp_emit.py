"""LP Form -> WASM emitter.

Compiles LP Form programs to WASM modules. Each procedure becomes a WASM
function. Variables are WASM locals. Multi-clause procedures use if-else
chains on guards. Tail calls use return_call.

Supports:
  - Global mutable i32 variables (gget/gset)
  - Mutable i32 GC arrays (aget/aset/anew)
  - Mutable funcref GC arrays (rnew) for WAM continuation stacks
  - Multi-clause dispatch with compound guards
  - Void (zero-output) procedures
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import (
    module, functype, export_func, export_global, I32,
    comptype_array, subtype, reftype, byte as enc_byte,
    global_entry, i32_const, ref_null,
    array_new_default, array_get, array_set,
)
from wir import WIR
from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst,
    GlobalDecl, ArrayDecl,
)


# Reserved names for trace instrumentation internals.
_TRACE_TMP_LOCAL = "__trace_tmp__"
_TRACE_BUF_NAME = "__trace_buf"
_TRACE_TOP_NAME = "__trace_top"
_TRACE_INIT_NAME = "__trace_init"
_TRACE_LEN_NAME = "__trace_len"
_TRACE_GET_NAME = "__trace_get"
_TRACE_RESET_NAME = "__trace_reset"


# PrimOp -> WIR method name (for binary arithmetic)
_ARITH_MAP = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div_s",
    "rem": "rem_s",
    "and": "and_",
    "or":  "or_",
}

# Guard op -> WIR method name
_GUARD_MAP = {
    "eq":  "eq",
    "ne":  "ne",
    "lt":  "lt_s",
    "le":  "le_s",
    "gt":  "gt_s",
    "ge":  "ge_s",
}


class LPEmitter:
    """Compile an LPProgram to WASM module bytes.

    Parameters
    ----------
    trace : bool
        When True, emit trace-buffer writes before each clause's body.
        Each trace record is: [size, proc_id, clause_idx, input_0, ...].
        Exports: __trace_init(), __trace_reset(), __trace_len() -> i32,
        __trace_get(idx: i32) -> i32, and global __trace_top.
    trace_size : int
        Capacity of the trace buffer in i32 cells (default 1M).
    """

    def __init__(self, trace: bool = False, trace_size: int = 1 << 20):
        self.trace = trace
        self.trace_size = trace_size

    def compile(self, program: LPProgram) -> bytes:
        self._program = program

        # --- Allocate WASM types ---
        self._types = []
        self._sig_to_typeidx = {}

        # Array types for declared arrays
        self._array_type_indices = {}  # array_name -> WASM type index
        for arr in program.arrays:
            if arr.kind == "i32":
                ti = len(self._types)
                self._types.append(
                    subtype(comptype_array((I32, True))))
            elif arr.kind == "ref":
                # funcref array: array of (ref null func)
                ft_idx = self._ensure_functype([], [])
                ti = len(self._types)
                self._types.append(
                    subtype(enc_byte(0x5E)
                            + reftype(True, ft_idx)
                            + enc_byte(0x01)))
            self._array_type_indices[arr.name] = ti

        # Trace buffer array type (i32 array)
        if self.trace:
            self._trace_array_ti = len(self._types)
            self._types.append(subtype(comptype_array((I32, True))))

        # --- Allocate WASM globals ---
        self._globals = []
        self._global_indices = {}  # global_name -> WASM global index

        # i32 globals from declarations
        for g in program.globals:
            gi = len(self._globals)
            self._globals.append(
                global_entry(I32, True, [i32_const(g.initial)]))
            self._global_indices[g.name] = gi

        # ref globals for declared arrays (initially null)
        self._array_global_indices = {}  # array_name -> WASM global index
        for arr in program.arrays:
            gi = len(self._globals)
            ti = self._array_type_indices[arr.name]
            self._globals.append(
                global_entry(reftype(True, ti), True, [ref_null(ti)]))
            self._array_global_indices[arr.name] = gi

        # Trace buffer globals: ref to the i32 array, plus write head
        if self.trace:
            self._trace_buf_gi = len(self._globals)
            self._globals.append(global_entry(
                reftype(True, self._trace_array_ti), True,
                [ref_null(self._trace_array_ti)]))
            self._trace_top_gi = len(self._globals)
            self._globals.append(global_entry(
                I32, True, [i32_const(0)]))

        # --- Build procedure index ---
        # Proc ids (used as trace record tags) start at 0 and match the
        # WASM function index for user procedures.
        self._proc_index = {}  # proc_name -> WASM function index
        for i, proc in enumerate(program.procedures):
            self._proc_index[proc.name] = i

        # --- Func type cache (initialized above) ---

        # --- Compile each procedure ---
        func_type_indices = []
        func_codes = []

        for proc in program.procedures:
            sig = (proc.arity_in, proc.arity_out)
            ti = self._ensure_functype(
                [I32] * sig[0], [I32] * sig[1])
            func_type_indices.append(ti)
            code = self._compile_proc(proc)
            func_codes.append(code)

        # --- Trace helper functions ---
        exports = []
        trace_init_idx = None
        if self.trace:
            trace_init_idx = len(func_type_indices)
            func_type_indices.append(self._ensure_functype([], []))
            func_codes.append(self._build_trace_init())
            exports.append(export_func(_TRACE_INIT_NAME, trace_init_idx))

            trace_reset_idx = len(func_type_indices)
            func_type_indices.append(self._ensure_functype([], []))
            func_codes.append(self._build_trace_reset())
            exports.append(export_func(_TRACE_RESET_NAME, trace_reset_idx))

            trace_len_idx = len(func_type_indices)
            func_type_indices.append(self._ensure_functype([], [I32]))
            func_codes.append(self._build_trace_len())
            exports.append(export_func(_TRACE_LEN_NAME, trace_len_idx))

            trace_get_idx = len(func_type_indices)
            func_type_indices.append(self._ensure_functype([I32], [I32]))
            func_codes.append(self._build_trace_get())
            exports.append(export_func(_TRACE_GET_NAME, trace_get_idx))

            exports.append(export_global(_TRACE_TOP_NAME, self._trace_top_gi))

        # --- Build entry wrapper if specified ---
        if program.entry:
            entry_proc = None
            for proc in program.procedures:
                if proc.name == program.entry:
                    entry_proc = proc
                    break

            if entry_proc:
                wrapper_ti = self._ensure_functype(
                    [I32] * entry_proc.arity_in,
                    [I32] * entry_proc.arity_out)
                params = [f"p{i}" for i in range(entry_proc.arity_in)]
                ir = WIR(params, results=[I32] * entry_proc.arity_out)
                # Auto-initialize the trace buffer on each top-level call so
                # repeated invocations start from a clean trace.
                if self.trace:
                    ir.fn_call(trace_init_idx)
                for p in params:
                    ir.local(p)
                ir.return_call(self._proc_index[program.entry])

                wrapper_idx = len(func_type_indices)
                func_type_indices.append(wrapper_ti)
                func_codes.append(ir.encode())
                exports.append(export_func("run", wrapper_idx))
        else:
            for proc in program.procedures:
                idx = self._proc_index[proc.name]
                exports.append(export_func(proc.name, idx))

        return module(
            types=self._types,
            funcs=func_type_indices,
            globals_=self._globals if self._globals else None,
            codes=func_codes,
            exports=exports,
        )

    def _ensure_functype(self, params, results):
        """Get or create a functype, return its type index."""
        key = (tuple(params), tuple(results))
        if key not in self._sig_to_typeidx:
            ti = len(self._types)
            self._types.append(functype(params, results))
            self._sig_to_typeidx[key] = ti
        return self._sig_to_typeidx[key]

    # -----------------------------------------------------------------
    # Procedure compilation
    # -----------------------------------------------------------------

    def _compile_proc(self, proc: LPProc) -> bytes:
        self._cur_proc = proc
        self._cur_proc_id = self._proc_index[proc.name]
        if len(proc.clauses) == 1:
            return self._compile_single_clause(proc, proc.clauses[0])
        else:
            return self._compile_multi_clause(proc)

    def _compile_single_clause(self, proc, clause):
        params = list(clause.head.inputs)
        ir = WIR(params, results=[I32] * proc.arity_out)

        if self.trace:
            ir.new_local(_TRACE_TMP_LOCAL)

        all_vars = self._collect_clause_vars(clause)
        for v in all_vars:
            if v not in clause.head.inputs:
                ir.new_local(v)

        var_map = {v: v for v in all_vars}

        if self.trace:
            self._emit_trace_append(
                ir, self._cur_proc_id, 0, clause.head.inputs, var_map)

        self._emit_goals_all(ir, clause.goals, clause.head, var_map)

        # Push outputs if last goal wasn't a tail call
        if not self._ends_with_tail_call(clause.goals):
            for out in clause.head.outputs:
                ir.local(var_map[out])

        return ir.encode()

    def _compile_multi_clause(self, proc):
        params = list(proc.clauses[0].head.inputs)
        ir = WIR(params, results=[I32] * proc.arity_out)

        if self.trace:
            ir.new_local(_TRACE_TMP_LOCAL)

        # Declare locals for all clauses (prefixed by clause index)
        clause_var_maps = []
        for ci, clause in enumerate(proc.clauses):
            var_map = {}
            for v in clause.head.inputs:
                var_map[v] = v
            all_vars = self._collect_clause_vars(clause)
            for v in all_vars:
                if v in clause.head.inputs:
                    var_map[v] = v
                else:
                    unique = f"_c{ci}_{v}"
                    var_map[v] = unique
                    ir.new_local(unique)
            clause_var_maps.append(var_map)

        # Block type for if-else that produces results
        result_bt = None
        if proc.arity_out > 0:
            result_bt = self._ensure_functype([], [I32] * proc.arity_out)

        # Guard temp locals (shared across clause dispatch)
        guard_local_counter = [0]

        def new_guard_local():
            name = f"_grd{guard_local_counter[0]}"
            guard_local_counter[0] += 1
            ir.new_local(name)
            return name

        self._emit_clause_chain(
            ir, proc, 0, clause_var_maps, result_bt, new_guard_local)

        return ir.encode()

    # -----------------------------------------------------------------
    # Multi-clause dispatch with compound guards
    # -----------------------------------------------------------------

    def _emit_clause_chain(self, ir, proc, idx, var_maps, result_bt,
                           new_guard_local):
        clause = proc.clauses[idx]
        var_map = var_maps[idx]
        is_last = (idx == len(proc.clauses) - 1)

        if is_last:
            # Last clause: no guard check, just execute everything.
            # Trace fires unconditionally since this clause is the fallback.
            if self.trace:
                self._emit_trace_append(
                    ir, self._cur_proc_id, idx,
                    clause.head.inputs, var_map)
            self._emit_goals_all(ir, clause.goals, clause.head, var_map)
            if not self._ends_with_tail_call(clause.goals):
                for out in clause.head.outputs:
                    ir.local(var_map[out])
            return

        # Split goals into preamble (up to last guard) and suffix
        last_guard_idx = -1
        for i, g in enumerate(clause.goals):
            if isinstance(g, Guard):
                last_guard_idx = i

        if last_guard_idx == -1:
            # No guards but not last clause — emit everything
            if self.trace:
                self._emit_trace_append(
                    ir, self._cur_proc_id, idx,
                    clause.head.inputs, var_map)
            self._emit_goals_all(ir, clause.goals, clause.head, var_map)
            if not self._ends_with_tail_call(clause.goals):
                for out in clause.head.outputs:
                    ir.local(var_map[out])
            return

        preamble = clause.goals[:last_guard_idx + 1]
        suffix = clause.goals[last_guard_idx + 1:]

        # Execute preamble: non-guard goals execute normally,
        # guard results go into temp locals
        guard_locals = []
        for goal in preamble:
            if isinstance(goal, Guard):
                gl = new_guard_local()
                self._emit_val(ir, goal.left, var_map)
                self._emit_val(ir, goal.right, var_map)
                getattr(ir, _GUARD_MAP[goal.op])()
                ir.set(gl)
                guard_locals.append(gl)
            else:
                self._emit_one_goal(ir, goal, var_map)

        # AND all guard conditions together
        ir.local(guard_locals[0])
        for gl in guard_locals[1:]:
            ir.local(gl)
            ir.and_()

        # if guards hold: trace + execute suffix + push outputs
        # else: try next clause
        with ir.if_else(result_bt) as ie:
            if self.trace:
                self._emit_trace_append(
                    ir, self._cur_proc_id, idx,
                    clause.head.inputs, var_map)
            for goal in suffix:
                self._emit_one_goal(ir, goal, var_map)
            if not self._ends_with_tail_call(suffix):
                for out in clause.head.outputs:
                    ir.local(var_map[out])

            ie.then_part()  # else branch

            self._emit_clause_chain(
                ir, proc, idx + 1, var_maps, result_bt, new_guard_local)

    # -----------------------------------------------------------------
    # Goal emission
    # -----------------------------------------------------------------

    def _emit_goals_all(self, ir, goals, head, var_map):
        """Emit all goals sequentially (for single-clause or last clause)."""
        for goal in goals:
            if isinstance(goal, Guard):
                # Guards in single/last clause: skip (always true)
                continue
            self._emit_one_goal(ir, goal, var_map)

    def _emit_one_goal(self, ir, goal, var_map):
        """Emit a single non-guard goal."""
        if isinstance(goal, PrimOp):
            self._emit_primop(ir, goal, var_map)
        elif isinstance(goal, Call):
            self._emit_call(ir, goal, var_map)
        elif isinstance(goal, Guard):
            pass  # guards handled by clause chain

    def _emit_primop(self, ir, op: PrimOp, var_map):
        if op.op == "copy":
            self._emit_val(ir, op.inputs[0], var_map)
            ir.set(var_map[op.outputs[0]])

        elif op.op == "gget":
            # gget(NAME; out) — read global
            name = op.inputs[0].name
            gi = self._global_indices[name]
            ir.gget(gi)
            ir.set(var_map[op.outputs[0]])

        elif op.op == "gset":
            # gset(NAME, val;) — write global
            name = op.inputs[0].name
            gi = self._global_indices[name]
            self._emit_val(ir, op.inputs[1], var_map)
            ir.gset(gi)

        elif op.op == "aget":
            # aget(ARR, idx; val) — read array element
            arr_name = op.inputs[0].name
            arr_gi = self._array_global_indices[arr_name]
            arr_ti = self._array_type_indices[arr_name]
            ir.gget(arr_gi)
            self._emit_val(ir, op.inputs[1], var_map)
            ir._emit(array_get(arr_ti))
            ir.set(var_map[op.outputs[0]])

        elif op.op == "aset":
            # aset(ARR, idx, val;) — write array element
            arr_name = op.inputs[0].name
            arr_gi = self._array_global_indices[arr_name]
            arr_ti = self._array_type_indices[arr_name]
            ir.gget(arr_gi)
            self._emit_val(ir, op.inputs[1], var_map)
            self._emit_val(ir, op.inputs[2], var_map)
            ir._emit(array_set(arr_ti))

        elif op.op == "anew":
            # anew(ARR, size;) — allocate i32 array
            arr_name = op.inputs[0].name
            arr_gi = self._array_global_indices[arr_name]
            arr_ti = self._array_type_indices[arr_name]
            self._emit_val(ir, op.inputs[1], var_map)
            ir._emit(array_new_default(arr_ti))
            ir.gset(arr_gi)

        elif op.op == "rnew":
            # rnew(ARR, size;) — allocate ref array
            arr_name = op.inputs[0].name
            arr_gi = self._array_global_indices[arr_name]
            arr_ti = self._array_type_indices[arr_name]
            self._emit_val(ir, op.inputs[1], var_map)
            ir._emit(array_new_default(arr_ti))
            ir.gset(arr_gi)

        elif op.op in _ARITH_MAP:
            for inp in op.inputs:
                self._emit_val(ir, inp, var_map)
            getattr(ir, _ARITH_MAP[op.op])()
            ir.set(var_map[op.outputs[0]])

        else:
            raise ValueError(f"unknown PrimOp: {op.op}")

    def _emit_call(self, ir, call: Call, var_map):
        fn_idx = self._proc_index[call.name]

        for inp in call.inputs:
            self._emit_val(ir, inp, var_map)

        if call.is_tail:
            ir.return_call(fn_idx)
        else:
            ir.fn_call(fn_idx)
            # Capture outputs (multi-value: first result deepest on stack)
            for out in reversed(call.outputs):
                ir.set(var_map[out])

    def _emit_val(self, ir, val, var_map):
        if isinstance(val, LPVar):
            ir.local(var_map[val.name])
        elif isinstance(val, LPConst):
            ir.const(val.value)
        else:
            raise ValueError(f"unexpected value: {val}")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _ends_with_tail_call(self, goals):
        if not goals:
            return False
        last = goals[-1]
        return isinstance(last, Call) and last.is_tail

    def _collect_clause_vars(self, clause):
        vs = set(clause.head.inputs)
        vs.update(clause.head.outputs)
        for goal in clause.goals:
            if isinstance(goal, Guard):
                if isinstance(goal.left, LPVar):
                    vs.add(goal.left.name)
                if isinstance(goal.right, LPVar):
                    vs.add(goal.right.name)
            elif isinstance(goal, (PrimOp, Call)):
                for inp in goal.inputs:
                    if isinstance(inp, LPVar):
                        vs.add(inp.name)
                vs.update(goal.outputs)
        return vs

    # -----------------------------------------------------------------
    # Trace instrumentation
    # -----------------------------------------------------------------

    def _emit_trace_append(self, ir, proc_id, clause_idx,
                           input_names, var_map):
        """Append a variable-length record to the trace buffer.

        Record layout at buf[top .. top+size]:
            buf[top+0]   = payload_size            (2 + arity_in)
            buf[top+1]   = proc_id
            buf[top+2]   = clause_idx
            buf[top+3+i] = input_i (for i in range(arity_in))
        Then top advances by 1 + payload_size.
        """
        payload_size = 2 + len(input_names)
        arr_ti = self._trace_array_ti
        buf_gi = self._trace_buf_gi
        top_gi = self._trace_top_gi

        # tmp = top
        ir.gget(top_gi)
        ir.set(_TRACE_TMP_LOCAL)

        # buf[tmp] = payload_size
        ir.gget(buf_gi)
        ir.local(_TRACE_TMP_LOCAL)
        ir.const(payload_size)
        ir._emit(array_set(arr_ti))

        # buf[tmp+1] = proc_id
        ir.gget(buf_gi)
        ir.local(_TRACE_TMP_LOCAL)
        ir.const(1)
        ir.add()
        ir.const(proc_id)
        ir._emit(array_set(arr_ti))

        # buf[tmp+2] = clause_idx
        ir.gget(buf_gi)
        ir.local(_TRACE_TMP_LOCAL)
        ir.const(2)
        ir.add()
        ir.const(clause_idx)
        ir._emit(array_set(arr_ti))

        for i, name in enumerate(input_names):
            ir.gget(buf_gi)
            ir.local(_TRACE_TMP_LOCAL)
            ir.const(3 + i)
            ir.add()
            ir.local(var_map[name])
            ir._emit(array_set(arr_ti))

        # top = tmp + 1 + payload_size
        ir.local(_TRACE_TMP_LOCAL)
        ir.const(1 + payload_size)
        ir.add()
        ir.gset(top_gi)

    def _build_trace_init(self):
        """Allocate a fresh trace buffer and reset the write head."""
        ir = WIR([], results=[])
        ir.const(self.trace_size)
        ir._emit(array_new_default(self._trace_array_ti))
        ir.gset(self._trace_buf_gi)
        ir.const(0)
        ir.gset(self._trace_top_gi)
        return ir.encode()

    def _build_trace_reset(self):
        """Reset the trace write head without reallocating the buffer."""
        ir = WIR([], results=[])
        ir.const(0)
        ir.gset(self._trace_top_gi)
        return ir.encode()

    def _build_trace_len(self):
        """Return the current write head (number of i32 cells written)."""
        ir = WIR([], results=[I32])
        ir.gget(self._trace_top_gi)
        return ir.encode()

    def _build_trace_get(self):
        """Read a single i32 cell from the trace buffer."""
        ir = WIR(['idx'], results=[I32])
        ir.gget(self._trace_buf_gi)
        ir.local('idx')
        ir._emit(array_get(self._trace_array_ti))
        return ir.encode()
