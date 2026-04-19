"""LP Form -> WASM emitter.

Compiles LP Form programs to WASM modules. Each procedure becomes a WASM
function. Variables are WASM locals. Multi-clause procedures use if-else
chains on guards. Tail calls use return_call.

No runtime needed — no heap, no unification, no choice points.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))

from encoder import (
    module, functype, export_func, I32,
)
from wir import WIR
from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst,
)


# PrimOp -> WIR method name
_PRIMOP_MAP = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div_s",
    "rem": "rem_s",
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
    """Compile an LPProgram to WASM module bytes."""

    def compile(self, program: LPProgram) -> bytes:
        # Build procedure index: name -> function index
        self._proc_index = {}
        for i, proc in enumerate(program.procedures):
            self._proc_index[proc.name] = i

        # Build types and function bodies
        types = []
        func_type_indices = []
        func_codes = []

        # One functype per unique signature
        sig_to_typeidx = {}
        for proc in program.procedures:
            sig = (proc.arity_in, proc.arity_out)
            if sig not in sig_to_typeidx:
                sig_to_typeidx[sig] = len(types)
                types.append(functype([I32] * sig[0], [I32] * sig[1]))

        # Compile each procedure
        for proc in program.procedures:
            sig = (proc.arity_in, proc.arity_out)
            func_type_indices.append(sig_to_typeidx[sig])
            code = self._compile_proc(proc, types, sig_to_typeidx)
            func_codes.append(code)

        # Build entry wrapper if specified
        exports = []
        if program.entry:
            entry_proc = None
            for proc in program.procedures:
                if proc.name == program.entry:
                    entry_proc = proc
                    break

            if entry_proc:
                # Create wrapper: (param i32)*arity_in -> (result i32)*arity_out
                # Just calls the entry procedure
                wrapper_sig = (entry_proc.arity_in, entry_proc.arity_out)
                wrapper_ti = sig_to_typeidx.get(wrapper_sig)
                if wrapper_ti is None:
                    wrapper_ti = len(types)
                    types.append(functype(
                        [I32] * wrapper_sig[0], [I32] * wrapper_sig[1]))

                params = [f"p{i}" for i in range(entry_proc.arity_in)]
                ir = WIR(params, results=[I32] * entry_proc.arity_out)
                for p in params:
                    ir.local(p)
                entry_fn = self._proc_index[program.entry]
                ir.return_call(entry_fn)

                wrapper_idx = len(func_type_indices)
                func_type_indices.append(wrapper_ti)
                func_codes.append(ir.encode())
                exports.append(export_func("run", wrapper_idx))
            else:
                # Entry not found, export nothing
                pass
        else:
            # No entry specified, export all procedures
            for proc in program.procedures:
                idx = self._proc_index[proc.name]
                exports.append(export_func(proc.name, idx))

        return module(
            types=types,
            funcs=func_type_indices,
            codes=func_codes,
            exports=exports,
        )

    def _compile_proc(self, proc: LPProc, types, sig_to_typeidx) -> bytes:
        """Compile a single procedure to WASM function body bytes."""
        if len(proc.clauses) == 1:
            return self._compile_single_clause(proc, proc.clauses[0])
        else:
            return self._compile_multi_clause(proc, types, sig_to_typeidx)

    def _compile_single_clause(self, proc: LPProc, clause: LPClause) -> bytes:
        """Compile a procedure with exactly one clause (no guards)."""
        params = list(clause.head.inputs)
        ir = WIR(params, results=[I32] * proc.arity_out)

        # Declare locals for all variables not in params
        all_vars = self._collect_clause_vars(clause)
        for v in all_vars:
            if v not in clause.head.inputs:
                ir.new_local(v)

        # Emit goals
        self._emit_goals(ir, clause.goals, clause.head)

        # If last goal is not a tail call, push outputs
        if not clause.goals or not (isinstance(clause.goals[-1], Call)
                                     and clause.goals[-1].is_tail):
            for out in clause.head.outputs:
                ir.local(out)

        return ir.encode()

    def _compile_multi_clause(self, proc: LPProc, types, sig_to_typeidx) -> bytes:
        """Compile a procedure with multiple clauses (guarded)."""
        params = list(proc.clauses[0].head.inputs)
        ir = WIR(params, results=[I32] * proc.arity_out)

        # Declare locals for all variables across all clauses
        # (each clause has its own namespace, but WASM locals are flat)
        # We prefix clause-local vars with clause index to avoid collisions
        clause_var_maps = []
        for ci, clause in enumerate(proc.clauses):
            var_map = {}
            for v in clause.head.inputs:
                var_map[v] = v  # inputs share the param locals
            all_vars = self._collect_clause_vars(clause)
            for v in all_vars:
                if v in clause.head.inputs:
                    var_map[v] = v
                else:
                    unique = f"_c{ci}_{v}"
                    var_map[v] = unique
                    ir.new_local(unique)
            clause_var_maps.append(var_map)

        # Need a block type for the if-else that produces results
        result_bt = None
        if proc.arity_out > 0:
            sig = (0, proc.arity_out)  # no params, just results
            if sig not in sig_to_typeidx:
                sig_to_typeidx[sig] = len(types)
                types.append(functype([], [I32] * proc.arity_out))
            result_bt = sig_to_typeidx[sig]

        # Emit nested if-else chain
        self._emit_clause_chain(ir, proc, 0, clause_var_maps, result_bt)

        return ir.encode()

    def _emit_clause_chain(self, ir, proc, idx, var_maps, result_bt):
        """Emit a chain of if-else for clauses starting at idx."""
        clause = proc.clauses[idx]
        var_map = var_maps[idx]
        is_last = (idx == len(proc.clauses) - 1)

        if is_last:
            # Last clause: no guard check needed, just emit body
            self._emit_goals_mapped(ir, clause.goals, clause.head, var_map)
            if not clause.goals or not (isinstance(clause.goals[-1], Call)
                                         and clause.goals[-1].is_tail):
                for out in clause.head.outputs:
                    ir.local(var_map[out])
            return

        # Find the guard(s) in this clause
        guards = [g for g in clause.goals if isinstance(g, Guard)]
        non_guards = [g for g in clause.goals if not isinstance(g, Guard)]

        if not guards:
            # No guard — this shouldn't happen in well-formed multi-clause
            # but handle gracefully: just emit this clause
            self._emit_goals_mapped(ir, clause.goals, clause.head, var_map)
            if not clause.goals or not (isinstance(clause.goals[-1], Call)
                                         and clause.goals[-1].is_tail):
                for out in clause.head.outputs:
                    ir.local(var_map[out])
            return

        # Emit guard condition
        guard = guards[0]
        self._emit_val(ir, guard.left, var_map)
        self._emit_val(ir, guard.right, var_map)
        getattr(ir, _GUARD_MAP[guard.op])()

        # If guard holds: execute this clause's body
        # Else: try next clause
        with ir.if_else(result_bt) as ie:
            self._emit_goals_mapped(ir, non_guards, clause.head, var_map)
            if not non_guards or not (isinstance(non_guards[-1], Call)
                                       and non_guards[-1].is_tail):
                for out in clause.head.outputs:
                    ir.local(var_map[out])

            ie.then_part()  # else branch

            self._emit_clause_chain(ir, proc, idx + 1, var_maps, result_bt)

    def _emit_goals(self, ir, goals, head):
        """Emit goals using identity variable mapping."""
        identity = {v: v for v in self._collect_clause_vars(
            LPClause(head=head, goals=goals))}
        for v in head.inputs:
            identity[v] = v
        self._emit_goals_mapped(ir, goals, head, identity)

    def _emit_goals_mapped(self, ir, goals, head, var_map):
        """Emit goals with a variable name mapping."""
        for goal in goals:
            if isinstance(goal, Guard):
                # Guards in single-clause procs are just assertions; skip
                continue
            elif isinstance(goal, PrimOp):
                self._emit_primop(ir, goal, var_map)
            elif isinstance(goal, Call):
                self._emit_call(ir, goal, var_map)

    def _emit_primop(self, ir, op: PrimOp, var_map):
        """Emit a primitive operation."""
        if op.op == "copy":
            # copy(src; dst) — just alias
            self._emit_val(ir, op.inputs[0], var_map)
            ir.set(var_map[op.outputs[0]])
        elif op.op in _PRIMOP_MAP:
            for inp in op.inputs:
                self._emit_val(ir, inp, var_map)
            getattr(ir, _PRIMOP_MAP[op.op])()
            # For standard binary ops: one output
            ir.set(var_map[op.outputs[0]])
        else:
            raise ValueError(f"unknown PrimOp: {op.op}")

    def _emit_call(self, ir, call: Call, var_map):
        """Emit a procedure call."""
        fn_idx = self._proc_index[call.name]

        # Push inputs
        for inp in call.inputs:
            self._emit_val(ir, inp, var_map)

        if call.is_tail:
            ir.return_call(fn_idx)
        else:
            ir.fn_call(fn_idx)
            # Capture outputs into locals (reverse order since WASM
            # multi-value results are on the stack in order)
            # Actually, WASM multi-value: first result is deepest on stack
            # So we need to set them in reverse order
            for out in reversed(call.outputs):
                ir.set(var_map[out])

    def _emit_val(self, ir, val, var_map):
        """Push a value onto the WASM stack."""
        if isinstance(val, LPVar):
            ir.local(var_map[val.name])
        elif isinstance(val, LPConst):
            ir.const(val.value)
        else:
            raise ValueError(f"unexpected value: {val}")

    def _collect_clause_vars(self, clause):
        """Collect all variable names used in a clause."""
        vs = set(clause.head.inputs)
        vs.update(clause.head.outputs)
        for goal in clause.goals:
            if isinstance(goal, Guard):
                if isinstance(goal.left, LPVar):
                    vs.add(goal.left.name)
                if isinstance(goal.right, LPVar):
                    vs.add(goal.right.name)
            elif isinstance(goal, PrimOp):
                for inp in goal.inputs:
                    if isinstance(inp, LPVar):
                        vs.add(inp.name)
                vs.update(goal.outputs)
            elif isinstance(goal, Call):
                for inp in goal.inputs:
                    if isinstance(inp, LPVar):
                        vs.add(inp.name)
                vs.update(goal.outputs)
        return vs
