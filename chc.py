"""CHC extraction from LP Form programs to SMT-LIB2.

LP Form IS Horn clauses. Each procedure becomes an uninterpreted predicate;
each clause becomes a Horn rule (forall vars, body => head).

State-passing transform (globals and arrays):
  - i32 globals become Int parameters (pre and post state)
  - i32 arrays become (Array Int Int) parameters (pre and post state)
  - ref (funcref) arrays become uninterpreted-sort parameters (pre/post).
    rget/rset/rnew operations are modelled as uninterpreted functions over
    the opaque sort — sound but no content reasoning.

Analysis determines which globals and arrays each procedure transitively
reads or writes, and threads only those through the predicate signature.

Predicate signature ordering:
    (proc args_in... state_pre_g... state_pre_a...
          args_out... state_post_g... state_post_a...)

Bitwise `and`/`or` PrimOps are modelled as uninterpreted functions
`lp_and`, `lp_or` (Z3 cannot reason about them semantically, but the
encoding stays sound).
"""

from lp_form import (LPProgram, LPProc, LPClause, LPHead,
                     PrimOp, Guard, Call, LPVar, LPConst)
import z3 as _z3


# LP guard op -> SMT-LIB2 formatter
_GUARD_FMT = {
    "eq": lambda l, r: f"(= {l} {r})",
    "ne": lambda l, r: f"(not (= {l} {r}))",
    "lt": lambda l, r: f"(< {l} {r})",
    "le": lambda l, r: f"(<= {l} {r})",
    "gt": lambda l, r: f"(> {l} {r})",
    "ge": lambda l, r: f"(>= {l} {r})",
}

# Binary arithmetic PrimOp -> SMT operator
_ARITH_BINOP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "div",
    "rem": "mod",
}


class CHCExtractor:
    def __init__(self, program: LPProgram):
        self.program = program
        self.globals_by_name = {g.name: g for g in program.globals}
        self.arrays_by_name = {a.name: a for a in program.arrays}
        self._uses_and = False
        self._uses_or = False
        self._analyze_state()

    # -- State analysis (transitive read/write per procedure) --

    def _analyze_state(self):
        self.uses_g = {p.name: set() for p in self.program.procedures}
        self.uses_a = {p.name: set() for p in self.program.procedures}
        calls = {p.name: set() for p in self.program.procedures}
        for proc in self.program.procedures:
            for clause in proc.clauses:
                for goal in clause.goals:
                    if isinstance(goal, PrimOp):
                        if goal.op in ("gget", "gset"):
                            self.uses_g[proc.name].add(goal.inputs[0].name)
                        elif goal.op in ("aget", "aset", "anew",
                                         "rget", "rset", "rnew"):
                            name = goal.inputs[0].name
                            if name in self.arrays_by_name:
                                self.uses_a[proc.name].add(name)
                        elif goal.op == "and":
                            self._uses_and = True
                        elif goal.op == "or":
                            self._uses_or = True
                    elif isinstance(goal, Call):
                        calls[proc.name].add(goal.name)

        changed = True
        while changed:
            changed = False
            for name in calls:
                for callee in calls[name]:
                    if callee not in self.uses_g:
                        continue
                    new_g = self.uses_g[callee] - self.uses_g[name]
                    new_a = self.uses_a[callee] - self.uses_a[name]
                    if new_g or new_a:
                        self.uses_g[name] |= new_g
                        self.uses_a[name] |= new_a
                        changed = True

    def _is_i32_array(self, name):
        arr = self.arrays_by_name.get(name)
        return arr is not None and arr.kind == "i32"

    def _state_for(self, proc_name):
        gs = sorted(self.uses_g.get(proc_name, set()))
        as_ = sorted(self.uses_a.get(proc_name, set()))
        return gs, as_

    # -- SMT sort helpers --

    def _sort_global(self, name):
        return "Int"

    def _sort_array(self, name):
        arr = self.arrays_by_name[name]
        if arr.kind == "i32":
            return "(Array Int Int)"
        else:
            return name  # opaque uninterpreted sort

    def _pred_signature(self, proc):
        gs, as_ = self._state_for(proc.name)
        types = ["Int"] * proc.arity_in
        types += [self._sort_global(g) for g in gs]
        types += [self._sort_array(a) for a in as_]
        types += ["Int"] * proc.arity_out
        types += [self._sort_global(g) for g in gs]
        types += [self._sort_array(a) for a in as_]
        return types

    # -- Top-level emission --

    def emit(self):
        lines = ["(set-logic HORN)"]

        # Declare opaque sorts for ref arrays
        for a in self.program.arrays:
            if a.kind != "i32":
                lines.append(f"(declare-sort {a.name} 0)")

        # Uninterpreted functions for ref array ops
        ref_arrays = {a.name for a in self.program.arrays if a.kind != "i32"}
        if ref_arrays:
            for name in sorted(ref_arrays):
                lines.append(
                    f"(declare-fun {name}_get ({name} Int) Int)")
                lines.append(
                    f"(declare-fun {name}_set ({name} Int Int) {name})")
                lines.append(
                    f"(declare-fun {name}_new (Int) {name})")

        if self._uses_and:
            lines.append("(declare-fun lp_and (Int Int) Int)")
        if self._uses_or:
            lines.append("(declare-fun lp_or (Int Int) Int)")

        for proc in self.program.procedures:
            types = self._pred_signature(proc)
            lines.append(
                f"(declare-fun {proc.name} ({' '.join(types)}) Bool)")

        for proc in self.program.procedures:
            for idx, clause in enumerate(proc.clauses):
                lines.append(self._emit_clause(proc, clause, idx))

        # Termination measure obligations
        for proc in self.program.procedures:
            if proc.measure:
                lines.append(self._emit_termination(proc))

        return "\n".join(lines) + "\n"

    # -- Termination measure obligations --

    def _emit_termination(self, proc):
        """Emit well-foundedness helper predicate for termination.

        Introduces a `_term_<name>` predicate with the same signature.
        Each clause maps to a rule where recursive calls to `proc` are
        replaced by calls to `_term_<name>`, plus the constraint that the
        measure of the recursive call is strictly less than the measure
        of the current call.  The base case (no recursion) maps to true.

        The query `(assert (=> (_term_<name> ...) false))` is NOT emitted —
        the caller adds their own.  If Spacer finds `sat`, the predicate
        is satisfiable (measure decreases at every step), meaning the
        procedure terminates.
        """
        gs, as_ = self._state_for(proc.name)
        measure_vars = proc.measure
        term_pred = f"_term_{proc.name}"
        lines = []

        # Declare the termination predicate
        types = self._pred_signature(proc)
        lines.append(f"(declare-fun {term_pred} ({' '.join(types)}) Bool)")

        for clause in proc.clauses:
            has_rec = any(isinstance(g, Call) and g.name == proc.name
                          for g in clause.goals)
            if not has_rec:
                # Base case: _term holds trivially (same shape as proc clause)
                self._emit_term_clause(proc, clause, term_pred, gs, as_,
                                       measure_vars, lines, is_base=True)
            else:
                self._emit_term_clause(proc, clause, term_pred, gs, as_,
                                       measure_vars, lines, is_base=False)

        return "\n".join(lines)

    def _emit_term_clause(self, proc, clause, term_pred, gs, as_,
                          measure_vars, lines, is_base):
        """Emit one clause of the _term_ predicate."""
        pre_g = {g: f"__{g}_in" for g in gs}
        pre_a = {a: f"__{a}_in" for a in as_}
        cur_g = dict(pre_g)
        cur_a = dict(pre_a)
        version_g = {g: 0 for g in gs}
        version_a = {a: 0 for a in as_}

        def fresh_g(name):
            version_g[name] += 1
            return f"__{name}_{version_g[name]}"

        def fresh_a(name):
            version_a[name] += 1
            return f"__{a}_{version_a[name]}"

        var_types = {}
        for v in clause.head.inputs:
            var_types[_smt_id(v)] = "Int"
        for g in gs:
            var_types[pre_g[g]] = self._sort_global(g)
        for a in as_:
            var_types[pre_a[a]] = self._sort_array(a)
        for v in clause.head.outputs:
            var_types.setdefault(_smt_id(v), "Int")

        head_input_map = {name: _smt_id(name) for name in clause.head.inputs}
        cur_measures = [head_input_map[mv] for mv in measure_vars if mv in head_input_map]

        constraints = []

        for goal in clause.goals:
            if isinstance(goal, Guard):
                l = self._val(goal.left)
                r = self._val(goal.right)
                constraints.append(_GUARD_FMT[goal.op](l, r))
            elif isinstance(goal, PrimOp):
                self._collect_primop(goal, var_types, cur_g, cur_a,
                                     pre_g, pre_a, fresh_g, fresh_a,
                                     constraints)
            elif isinstance(goal, Call):
                if goal.name == proc.name:
                    # Recursive call → use _term_pred + measure decrease
                    rec_measures = [self._val(v) for v in goal.inputs
                                    if isinstance(v, LPVar)]
                    c_gs, c_as = gs, as_
                    in_args = [self._val(v) for v in goal.inputs]
                    pre_g_args = [cur_g[g] for g in c_gs]
                    pre_a_args = [cur_a[a] for a in c_as]
                    out_args = [_smt_id(o) for o in goal.outputs]
                    for o in out_args:
                        var_types.setdefault(o, "Int")
                    post_g_args = []
                    for g in c_gs:
                        n = fresh_g(g)
                        var_types[n] = self._sort_global(g)
                        post_g_args.append(n)
                        cur_g[g] = n
                    post_a_args = []
                    for a in c_as:
                        n = fresh_a(a)
                        var_types[n] = self._sort_array(a)
                        post_a_args.append(n)
                        cur_a[a] = n
                    all_args = (in_args + pre_g_args + pre_a_args
                                + out_args + post_g_args + post_a_args)
                    constraints.append(f"({term_pred} {' '.join(all_args)})")
                    # Measure decrease
                    for cm, rm in zip(cur_measures, rec_measures):
                        if rm != cm:
                            constraints.append(f"(< {rm} {cm})")
                else:
                    self._collect_call(goal, var_types, cur_g, cur_a,
                                       fresh_g, fresh_a, constraints)

        # Build head term
        head_args = [_smt_id(v) for v in clause.head.inputs]
        head_args += [pre_g[g] for g in gs]
        head_args += [pre_a[a] for a in as_]
        head_args += [_smt_id(v) for v in clause.head.outputs]
        head_args += [cur_g[g] for g in gs]
        head_args += [cur_a[a] for a in as_]

        head_str = f"({term_pred} {' '.join(head_args)})" if head_args \
            else f"({term_pred})"

        if not constraints:
            body = "true"
        elif len(constraints) == 1:
            body = constraints[0]
        else:
            body = "(and " + " ".join(constraints) + ")"

        qvars = sorted(var_types.keys())
        quant = " ".join(f"({v} {var_types[v]})" for v in qvars)

        if quant:
            lines.append(f"(assert (forall ({quant}) (=> {body} {head_str})))")
        else:
            lines.append(f"(assert (=> {body} {head_str}))")

    def _collect_primop(self, goal, var_types, cur_g, cur_a,
                        pre_g, pre_a, fresh_g, fresh_a, constraints):
        """Collect a PrimOp constraint into `constraints`."""
        op = goal.op
        if op in _ARITH_BINOP:
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            a, b = self._val(goal.inputs[0]), self._val(goal.inputs[1])
            constraints.append(f"(= {out} ({_ARITH_BINOP[op]} {a} {b}))")
        elif op == "copy":
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            constraints.append(f"(= {out} {self._val(goal.inputs[0])})")
        elif op == "neg":
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            constraints.append(f"(= {out} (- {self._val(goal.inputs[0])}))")
        elif op == "mod":
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            a, b = self._val(goal.inputs[0]), self._val(goal.inputs[1])
            constraints.append(f"(= {out} (mod {a} {b}))")
        elif op == "and":
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            a, b = self._val(goal.inputs[0]), self._val(goal.inputs[1])
            constraints.append(f"(= {out} (lp_and {a} {b}))")
        elif op == "or":
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            a, b = self._val(goal.inputs[0]), self._val(goal.inputs[1])
            constraints.append(f"(= {out} (lp_or {a} {b}))")
        elif op == "gget":
            gname = goal.inputs[0].name
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            constraints.append(f"(= {out} {cur_g[gname]})")
        elif op == "gset":
            gname = goal.inputs[0].name
            val = self._val(goal.inputs[1])
            new = fresh_g(gname)
            constraints.append(f"(= {new} {val})")
            cur_g[gname] = new
        elif op == "aget":
            aname = goal.inputs[0].name
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            if self._is_i32_array(aname):
                idx = self._val(goal.inputs[1])
                constraints.append(f"(= {out} (select {cur_a[aname]} {idx}))")
            else:
                # ref array: uninterpreted read
                idx = self._val(goal.inputs[1])
                constraints.append(
                    f"(= {out} ({aname}_get {cur_a[aname]} {idx}))")
        elif op == "aset":
            aname = goal.inputs[0].name
            idx = self._val(goal.inputs[1])
            val = self._val(goal.inputs[2])
            new = fresh_a(aname)
            var_types[new] = self._sort_array(aname)
            if self._is_i32_array(aname):
                constraints.append(f"(= {new} (store {cur_a[aname]} {idx} {val}))")
            else:
                constraints.append(f"(= {new} ({aname}_set {cur_a[aname]} {idx} {val}))")
            cur_a[aname] = new
        elif op == "anew":
            aname = goal.inputs[0].name
            new = fresh_a(aname)
            var_types[new] = self._sort_array(aname)
            size = self._val(goal.inputs[1]) if len(goal.inputs) > 1 else "0"
            if self._is_i32_array(aname):
                cur_a[aname] = new  # leave unconstrained
            else:
                constraints.append(f"(= {new} ({aname}_new {size}))")
                cur_a[aname] = new
        elif op == "rget":
            aname = goal.inputs[0].name
            out = _smt_id(goal.outputs[0])
            var_types[out] = "Int"
            idx = self._val(goal.inputs[1])
            constraints.append(f"(= {out} ({aname}_get {cur_a[aname]} {idx}))")
        elif op == "rset":
            aname = goal.inputs[0].name
            idx = self._val(goal.inputs[1])
            val = self._val(goal.inputs[2])
            new = fresh_a(aname)
            var_types[new] = self._sort_array(aname)
            constraints.append(f"(= {new} ({aname}_set {cur_a[aname]} {idx} {val}))")
            cur_a[aname] = new
        elif op == "rnew":
            aname = goal.inputs[0].name
            new = fresh_a(aname)
            var_types[new] = self._sort_array(aname)
            size = self._val(goal.inputs[1]) if len(goal.inputs) > 1 else "0"
            constraints.append(f"(= {new} ({aname}_new {size}))")
            cur_a[aname] = new

    def _collect_call(self, goal, var_types, cur_g, cur_a,
                      fresh_g, fresh_a, constraints):
        """Collect a Call constraint into `constraints`."""
        callee = goal.name
        c_gs, c_as = self._state_for(callee)
        in_args = [self._val(v) for v in goal.inputs]
        pre_g_args = [cur_g[g] for g in c_gs]
        pre_a_args = [cur_a[a] for a in c_as]
        out_args = [_smt_id(o) for o in goal.outputs]
        for o in out_args:
            var_types.setdefault(o, "Int")
        post_g_args = []
        for g in c_gs:
            n = fresh_g(g)
            var_types[n] = self._sort_global(g)
            post_g_args.append(n)
            cur_g[g] = n
        post_a_args = []
        for a in c_as:
            n = fresh_a(a)
            var_types[n] = self._sort_array(a)
            post_a_args.append(n)
            cur_a[a] = n
        all_args = (in_args + pre_g_args + pre_a_args
                    + out_args + post_g_args + post_a_args)
        constraints.append(f"({callee} {' '.join(all_args)})")

    # -- Prior-clause negation (first-match-wins ordering) --

    def _emit_prior_negation(self, prior, prior_idx, current_head,
                             pre_g, pre_a, var_types):
        """Encode 'prior clause did not match' for current clause body.

        Processes `prior.goals` up to the last Guard (the preamble that
        determines whether the clause matches), renaming locals with a
        per-clause prefix. Prior-head input names are remapped positionally
        to the current clause's head input names — LP Form permits
        different clauses of a procedure to use different variable names.

        Preambles must be PURE (no state mutation, no calls); this is the
        common case and matches how lp_emit.py dispatches.
        """
        last_guard_idx = -1
        for i, g in enumerate(prior.goals):
            if isinstance(g, Guard):
                last_guard_idx = i
        if last_guard_idx == -1:
            # Prior has no guards — always matches at runtime, so any
            # clause after it is unreachable. Encode as `false` so the
            # Horn rule for the current clause is vacuously true.
            return "false"

        preamble = prior.goals[:last_guard_idx + 1]

        input_rename = {
            prior_name: current_name
            for prior_name, current_name
            in zip(prior.head.inputs, current_head.inputs)
        }

        prefix = f"__c{prior_idx}_"
        local_rename = {}

        # Per-negation state version tracking (independent of main clause's
        # state — since this is a hypothetical execution of prior preamble).
        cur_g = dict(pre_g)
        cur_a = dict(pre_a)
        version_g = {g: 0 for g in pre_g}
        version_a = {a: 0 for a in pre_a}

        def fresh_g(name):
            version_g[name] += 1
            n = f"{prefix}{name}_{version_g[name]}"
            var_types[n] = self._sort_global(name)
            return n

        def fresh_a(name):
            version_a[name] += 1
            n = f"{prefix}{name}_{version_a[name]}"
            var_types[n] = self._sort_array(name)
            return n

        def rename(name):
            if name in input_rename:
                return _smt_id(input_rename[name])
            if name not in local_rename:
                local_rename[name] = prefix + _smt_id(name)
                var_types[local_rename[name]] = "Int"
            return local_rename[name]

        def val_for(v):
            if isinstance(v, LPVar):
                return rename(v.name)
            if isinstance(v, LPConst):
                if v.value < 0:
                    return f"(- {-v.value})"
                return str(v.value)
            raise ValueError(f"unknown value: {v!r}")

        def_parts = []   # constraints that always hold (reads, call facts)
        guards = []      # guard constraints (we'll negate their conjunction)

        for goal in preamble:
            if isinstance(goal, Guard):
                l = val_for(goal.left)
                r = val_for(goal.right)
                guards.append(_GUARD_FMT[goal.op](l, r))
                continue

            if isinstance(goal, Call):
                callee = goal.name
                c_gs, c_as = self._state_for(callee)
                in_args = [val_for(v) for v in goal.inputs]
                pre_g_args = [cur_g[g] for g in c_gs]
                pre_a_args = [cur_a[a] for a in c_as]
                out_args = [rename(o) for o in goal.outputs]
                post_g_args = []
                for g in c_gs:
                    n = fresh_g(g)
                    post_g_args.append(n)
                    cur_g[g] = n
                post_a_args = []
                for a in c_as:
                    n = fresh_a(a)
                    post_a_args.append(n)
                    cur_a[a] = n
                all_args = (in_args + pre_g_args + pre_a_args +
                            out_args + post_g_args + post_a_args)
                if all_args:
                    def_parts.append(f"({callee} {' '.join(all_args)})")
                else:
                    def_parts.append(f"({callee})")
                continue

            if not isinstance(goal, PrimOp):
                raise ValueError(f"unknown goal in preamble: {goal!r}")

            op = goal.op
            if op in _ARITH_BINOP:
                out = rename(goal.outputs[0])
                a = val_for(goal.inputs[0])
                b = val_for(goal.inputs[1])
                def_parts.append(f"(= {out} ({_ARITH_BINOP[op]} {a} {b}))")
            elif op == "copy":
                out = rename(goal.outputs[0])
                def_parts.append(f"(= {out} {val_for(goal.inputs[0])})")
            elif op == "neg":
                out = rename(goal.outputs[0])
                def_parts.append(f"(= {out} (- {val_for(goal.inputs[0])}))")
            elif op == "mod":
                out = rename(goal.outputs[0])
                a = val_for(goal.inputs[0]); b = val_for(goal.inputs[1])
                def_parts.append(f"(= {out} (mod {a} {b}))")
            elif op == "and":
                out = rename(goal.outputs[0])
                a = val_for(goal.inputs[0]); b = val_for(goal.inputs[1])
                def_parts.append(f"(= {out} (lp_and {a} {b}))")
            elif op == "or":
                out = rename(goal.outputs[0])
                a = val_for(goal.inputs[0]); b = val_for(goal.inputs[1])
                def_parts.append(f"(= {out} (lp_or {a} {b}))")
            elif op == "gget":
                gname = goal.inputs[0].name
                out = rename(goal.outputs[0])
                def_parts.append(f"(= {out} {cur_g[gname]})")
            elif op == "gset":
                gname = goal.inputs[0].name
                v = val_for(goal.inputs[1])
                n = fresh_g(gname)
                def_parts.append(f"(= {n} {v})")
                cur_g[gname] = n
            elif op == "aget":
                aname = goal.inputs[0].name
                out = rename(goal.outputs[0])
                if self._is_i32_array(aname):
                    idx = val_for(goal.inputs[1])
                    def_parts.append(f"(= {out} (select {cur_a[aname]} {idx}))")
                # ref-array read: leave `out` unconstrained
            elif op == "aset":
                aname = goal.inputs[0].name
                if self._is_i32_array(aname):
                    idx = val_for(goal.inputs[1])
                    v = val_for(goal.inputs[2])
                    n = fresh_a(aname)
                    def_parts.append(f"(= {n} (store {cur_a[aname]} {idx} {v}))")
                    cur_a[aname] = n
            elif op == "anew":
                aname = goal.inputs[0].name
                n = fresh_a(aname)
                size = val_for(goal.inputs[1]) if len(goal.inputs) > 1 else "0"
                if self._is_i32_array(aname):
                    cur_a[aname] = n
                else:
                    def_parts.append(f"(= {n} ({aname}_new {size}))")
                    cur_a[aname] = n
            elif op == "rget":
                aname = goal.inputs[0].name
                out = rename(goal.outputs[0])
                idx = val_for(goal.inputs[1])
                def_parts.append(f"(= {out} ({aname}_get {cur_a[aname]} {idx}))")
            elif op == "rset":
                aname = goal.inputs[0].name
                idx = val_for(goal.inputs[1])
                v = val_for(goal.inputs[2])
                n = fresh_a(aname)
                def_parts.append(f"(= {n} ({aname}_set {cur_a[aname]} {idx} {v}))")
                cur_a[aname] = n
            elif op == "rnew":
                aname = goal.inputs[0].name
                n = fresh_a(aname)
                size = val_for(goal.inputs[1]) if len(goal.inputs) > 1 else "0"
                def_parts.append(f"(= {n} ({aname}_new {size}))")
                cur_a[aname] = n
            else:
                raise NotImplementedError(
                    f"PrimOp {op!r} not supported in preamble negation")

        if not guards:
            return "false"

        neg_guards = (
            f"(not {guards[0]})" if len(guards) == 1
            else "(not (and " + " ".join(guards) + "))"
        )
        if not def_parts:
            return neg_guards
        return "(and " + " ".join(def_parts + [neg_guards]) + ")"

    # -- Clause emission --

    def _emit_clause(self, proc, clause, clause_idx):
        head = clause.head
        gs, as_ = self._state_for(proc.name)

        # Pre-state entry names
        pre_g = {g: f"__{g}_in" for g in gs}
        pre_a = {a: f"__{a}_in" for a in as_}

        # Current state — starts at pre, updated after each mutation
        cur_g = dict(pre_g)
        cur_a = dict(pre_a)

        version_g = {g: 0 for g in gs}
        version_a = {a: 0 for a in as_}

        def fresh_g(name):
            version_g[name] += 1
            return f"__{name}_{version_g[name]}"

        def fresh_a(name):
            version_a[name] += 1
            return f"__{name}_{version_a[name]}"

        # var -> SMT sort (for forall quantifier)
        var_types = {}
        for v in head.inputs:
            var_types[_smt_id(v)] = "Int"
        for g in gs:
            var_types[pre_g[g]] = self._sort_global(g)
        for a in as_:
            var_types[pre_a[a]] = self._sort_array(a)
        for v in head.outputs:
            var_types.setdefault(_smt_id(v), "Int")

        constraints = []

        # Clause-ordering semantics: LP Form dispatches first-match-wins.
        # Clause k fires iff clauses 0..k-1 all FAILED to match. We encode
        # this by adding, for each prior clause, the constraint that its
        # preamble guards DID NOT all hold. Prior clauses' preamble reads
        # are also folded in as equality constraints (they're functional
        # in the pre-state).
        for prior_idx in range(clause_idx):
            neg = self._emit_prior_negation(
                proc.clauses[prior_idx], prior_idx, head, pre_g, pre_a,
                var_types)
            if neg is not None:
                constraints.append(neg)

        for goal in clause.goals:
            if isinstance(goal, Guard):
                l = self._val(goal.left)
                r = self._val(goal.right)
                constraints.append(_GUARD_FMT[goal.op](l, r))

            elif isinstance(goal, PrimOp):
                op = goal.op
                if op in _ARITH_BINOP:
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    a = self._val(goal.inputs[0])
                    b = self._val(goal.inputs[1])
                    constraints.append(
                        f"(= {out} ({_ARITH_BINOP[op]} {a} {b}))")
                elif op == "copy":
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    constraints.append(f"(= {out} {self._val(goal.inputs[0])})")
                elif op == "neg":
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    constraints.append(
                        f"(= {out} (- {self._val(goal.inputs[0])}))")
                elif op == "mod":
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    a = self._val(goal.inputs[0])
                    b = self._val(goal.inputs[1])
                    constraints.append(f"(= {out} (mod {a} {b}))")
                elif op == "and":
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    a = self._val(goal.inputs[0])
                    b = self._val(goal.inputs[1])
                    constraints.append(f"(= {out} (lp_and {a} {b}))")
                elif op == "or":
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    a = self._val(goal.inputs[0])
                    b = self._val(goal.inputs[1])
                    constraints.append(f"(= {out} (lp_or {a} {b}))")
                elif op == "gget":
                    gname = goal.inputs[0].name
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    constraints.append(f"(= {out} {cur_g[gname]})")
                elif op == "gset":
                    gname = goal.inputs[0].name
                    val = self._val(goal.inputs[1])
                    new = fresh_g(gname)
                    var_types[new] = self._sort_global(gname)
                    constraints.append(f"(= {new} {val})")
                    cur_g[gname] = new
                elif op == "aget":
                    aname = goal.inputs[0].name
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    if self._is_i32_array(aname):
                        idx = self._val(goal.inputs[1])
                        constraints.append(
                            f"(= {out} (select {cur_a[aname]} {idx}))")
                    else:
                        idx = self._val(goal.inputs[1])
                        constraints.append(
                            f"(= {out} ({aname}_get {cur_a[aname]} {idx}))")
                elif op == "aset":
                    aname = goal.inputs[0].name
                    idx = self._val(goal.inputs[1])
                    val = self._val(goal.inputs[2])
                    new = fresh_a(aname)
                    var_types[new] = self._sort_array(aname)
                    if self._is_i32_array(aname):
                        constraints.append(
                            f"(= {new} (store {cur_a[aname]} {idx} {val}))")
                    else:
                        constraints.append(
                            f"(= {new} ({aname}_set {cur_a[aname]} {idx} {val}))")
                    cur_a[aname] = new
                elif op == "anew":
                    aname = goal.inputs[0].name
                    new = fresh_a(aname)
                    var_types[new] = self._sort_array(aname)
                    size = (self._val(goal.inputs[1])
                            if len(goal.inputs) > 1 else "0")
                    if self._is_i32_array(aname):
                        cur_a[aname] = new
                    else:
                        constraints.append(f"(= {new} ({aname}_new {size}))")
                        cur_a[aname] = new
                elif op == "rget":
                    aname = goal.inputs[0].name
                    out = _smt_id(goal.outputs[0])
                    var_types[out] = "Int"
                    idx = self._val(goal.inputs[1])
                    constraints.append(
                        f"(= {out} ({aname}_get {cur_a[aname]} {idx}))")
                elif op == "rset":
                    aname = goal.inputs[0].name
                    idx = self._val(goal.inputs[1])
                    val = self._val(goal.inputs[2])
                    new = fresh_a(aname)
                    var_types[new] = self._sort_array(aname)
                    constraints.append(
                        f"(= {new} ({aname}_set {cur_a[aname]} {idx} {val}))")
                    cur_a[aname] = new
                elif op == "rnew":
                    aname = goal.inputs[0].name
                    new = fresh_a(aname)
                    var_types[new] = self._sort_array(aname)
                    size = (self._val(goal.inputs[1])
                            if len(goal.inputs) > 1 else "0")
                    constraints.append(f"(= {new} ({aname}_new {size}))")
                    cur_a[aname] = new
                else:
                    raise NotImplementedError(
                        f"PrimOp {op!r} not supported in CHC extraction")

            elif isinstance(goal, Call):
                callee = goal.name
                c_gs, c_as = self._state_for(callee)

                in_args = [self._val(v) for v in goal.inputs]

                pre_g_args = []
                for g in c_gs:
                    if g not in cur_g:
                        raise RuntimeError(
                            f"caller {proc.name} missing global {g} "
                            f"required by callee {callee}")
                    pre_g_args.append(cur_g[g])
                pre_a_args = []
                for a in c_as:
                    if a not in cur_a:
                        raise RuntimeError(
                            f"caller {proc.name} missing array {a} "
                            f"required by callee {callee}")
                    pre_a_args.append(cur_a[a])

                out_args = [_smt_id(o) for o in goal.outputs]
                for o in out_args:
                    var_types[o] = "Int"

                post_g_args = []
                for g in c_gs:
                    n = fresh_g(g)
                    var_types[n] = self._sort_global(g)
                    post_g_args.append(n)
                    cur_g[g] = n
                post_a_args = []
                for a in c_as:
                    n = fresh_a(a)
                    var_types[n] = self._sort_array(a)
                    post_a_args.append(n)
                    cur_a[a] = n

                all_args = (in_args + pre_g_args + pre_a_args +
                            out_args + post_g_args + post_a_args)
                if all_args:
                    constraints.append(f"({callee} {' '.join(all_args)})")
                else:
                    constraints.append(f"({callee})")

            else:
                raise ValueError(f"unknown goal: {goal!r}")

        # Build head term
        head_args = [_smt_id(v) for v in head.inputs]
        head_args += [pre_g[g] for g in gs]
        head_args += [pre_a[a] for a in as_]
        head_args += [_smt_id(v) for v in head.outputs]
        head_args += [cur_g[g] for g in gs]
        head_args += [cur_a[a] for a in as_]

        if head_args:
            head_str = f"({proc.name} {' '.join(head_args)})"
        else:
            head_str = f"({proc.name})"

        if not constraints:
            body = "true"
        elif len(constraints) == 1:
            body = constraints[0]
        else:
            body = "(and " + " ".join(constraints) + ")"

        qvars = sorted(var_types.keys())
        quant = " ".join(f"({v} {var_types[v]})" for v in qvars)

        if quant:
            return f"(assert (forall ({quant}) (=> {body} {head_str})))"
        return f"(assert (=> {body} {head_str}))"

    def _val(self, v):
        if isinstance(v, LPVar):
            return _smt_id(v.name)
        if isinstance(v, LPConst):
            if v.value < 0:
                return f"(- {-v.value})"
            return str(v.value)
        raise ValueError(f"unknown value: {v!r}")


# SMT-LIB2 simple symbols disallow apostrophes. LP Form variable names
# like `n'`, `i'`, `b_prime'` are common, so substitute a safe sequence.
def _smt_id(name: str) -> str:
    return name.replace("'", "_p")


def _transitive_callees(program: LPProgram, entries) -> set:
    """Return the set of procedure names reachable from `entries`."""
    proc_by_name = {p.name: p for p in program.procedures}
    reachable = set(entries)
    frontier = list(entries)
    while frontier:
        name = frontier.pop()
        proc = proc_by_name.get(name)
        if proc is None:
            continue
        for clause in proc.clauses:
            for goal in clause.goals:
                if isinstance(goal, Call) and goal.name not in reachable:
                    reachable.add(goal.name)
                    frontier.append(goal.name)
    return reachable


def check_property(smt_source, query, timeout_ms=30_000):
    """Check a property against a CHC system. Returns (result, cex).

    result is one of: "holds", "violated", "unknown".
    cex is None unless result == "violated", in which case it is a dict
    mapping variable names to values from the counterexample model.
    """
    full = smt_source + query
    s = _z3.SolverFor("HORN")
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.from_string(full)
    r = s.check()

    if r == _z3.sat:
        return ("holds", None)
    elif r == _z3.unsat:
        cex = _extract_cex(s)
        return ("violated", cex)
    else:
        return ("unknown", None)


def _extract_cex(solver):
    """Try to extract a counterexample from an unsat HORN solver.

    For Spacer, after unsat the solver may have a model available via
    s.model().  Walk the model and return a dict of variable -> value.
    Returns None if no model is available.
    """
    try:
        m = solver.model()
        if m is None:
            return None
        cex = {}
        for decl in m.decls():
            name = str(decl.name())
            val = m.get_interp(decl)
            if val is not None:
                try:
                    cex[name] = val.as_long()
                except (AttributeError, ValueError):
                    cex[name] = str(val)
            else:
                cex[name] = str(m.eval(decl()))
        return cex if cex else None
    except Exception:
        return None


def extract_chc(program: LPProgram, slice_to=None) -> str:
    """Extract CHC from an LP Form program; return SMT-LIB2 source.

    If `slice_to` is given (an iterable of proc names), the output is
    restricted to those procedures plus their transitive callees.
    Spacer handles small sliced programs much better than the full
    runtime, so property tests that target a single predicate should
    slice aggressively.
    """
    if slice_to is not None:
        keep = _transitive_callees(program, list(slice_to))
        program = LPProgram(
            procedures=[p for p in program.procedures if p.name in keep],
            globals=program.globals,
            arrays=program.arrays,
            entry=program.entry,
        )
    return CHCExtractor(program).emit()
