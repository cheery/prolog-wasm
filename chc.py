"""CHC extraction from LP Form programs to SMT-LIB2.

LP Form IS Horn clauses. Each procedure becomes an uninterpreted predicate;
each clause becomes a Horn rule (forall vars, body => head).

State-passing transform (globals and arrays):
  - i32 globals become Int parameters (pre and post state)
  - i32 arrays become (Array Int Int) parameters (pre and post state)
  - ref (funcref) arrays are abstracted away — rget/rset/rnew become
    no-ops and ref arrays are not threaded through predicates.
    This is sound but loses information about those arrays.

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
                        elif goal.op in ("aget", "aset", "anew"):
                            name = goal.inputs[0].name
                            if self._is_i32_array(name):
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
        if arr.kind != "i32":
            raise NotImplementedError(
                f"array {name!r} has kind {arr.kind!r}; CHC only supports i32")
        return "(Array Int Int)"

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

        if self._uses_and:
            lines.append("(declare-fun lp_and (Int Int) Int)")
        if self._uses_or:
            lines.append("(declare-fun lp_or (Int Int) Int)")

        for proc in self.program.procedures:
            types = self._pred_signature(proc)
            lines.append(
                f"(declare-fun {proc.name} ({' '.join(types)}) Bool)")

        for proc in self.program.procedures:
            for clause in proc.clauses:
                lines.append(self._emit_clause(proc, clause))

        return "\n".join(lines) + "\n"

    # -- Clause emission --

    def _emit_clause(self, proc, clause):
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
                    # else: ref array read — leave out unconstrained (no-op)
                elif op == "aset":
                    aname = goal.inputs[0].name
                    if self._is_i32_array(aname):
                        idx = self._val(goal.inputs[1])
                        val = self._val(goal.inputs[2])
                        new = fresh_a(aname)
                        var_types[new] = self._sort_array(aname)
                        constraints.append(
                            f"(= {new} (store {cur_a[aname]} {idx} {val}))")
                        cur_a[aname] = new
                    # else: ref array write — no-op
                elif op == "anew":
                    aname = goal.inputs[0].name
                    if self._is_i32_array(aname):
                        new = fresh_a(aname)
                        var_types[new] = self._sort_array(aname)
                        cur_a[aname] = new
                    # else: ref array alloc — no-op
                elif op in ("rget", "rset", "rnew"):
                    # ref (funcref) array operations are abstracted out.
                    # rget has an output (funcref) that we simply leave
                    # unbound — but LP Form stores it as an output name,
                    # so we must still declare it if present.
                    for o in goal.outputs:
                        var_types[_smt_id(o)] = "Int"
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


def extract_chc(program: LPProgram) -> str:
    """Extract CHC from an LP Form program; return SMT-LIB2 source."""
    return CHCExtractor(program).emit()
