"""Elaboration pass for LP Form type system.

Desugars high-level type features into the core LP Form IR:
  - Constructor patterns in call outputs -> tag guards + struct_get PrimOps
  - Field access values (expr.field) -> struct_get PrimOps with temporaries
  - struct_new/struct_get PrimOps get type metadata for the emitter

After elaboration, the program contains only core IR nodes (PrimOp, Guard, Call)
with struct_new/struct_get PrimOps carrying type metadata. No LPPattern or
LPFieldAccess nodes remain.
"""

from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst, LPFieldAccess, LPPattern,
    LPStructDecl, LPConstructor, LPSumDecl,
    flatten_outputs,
)


class TypeEnv:
    """Type environment built from struct and sum declarations."""

    def __init__(self, program):
        self.structs = {}    # name -> LPStructDecl
        self.sums = {}       # name -> LPSumDecl
        self.ctors = {}      # ctor_name -> (LPSumDecl, LPConstructor, tag_index)
        self._layouts = {}   # type_name -> list of (field_name, field_type)
        self._proc_output_types = {}  # proc_name -> list[type_name|None]

        for s in program.structs:
            self.structs[s.name] = s
            self._layouts[s.name] = s.fields

        for s in program.sums:
            self.sums[s.name] = s
            self._layouts[s.name] = self._compute_sum_layout(s)
            for i, ctor in enumerate(s.constructors):
                self.ctors[ctor.name] = (s, ctor, i)

        for p in program.procedures:
            if p.output_types is not None:
                self._proc_output_types[p.name] = p.output_types

    @staticmethod
    def _compute_sum_layout(sum_decl):
        """Compute struct field layout for a sum type.

        Layout: [(__tag, i32), (_f0, type0), (_f1, type1), ...]
        Payload fields are sized for the largest constructor.
        """
        max_params = max(len(c.params) for c in sum_decl.constructors)
        fields = [("__tag", "i32")]
        for i in range(max_params):
            ftype = "i32"
            for c in sum_decl.constructors:
                if i < len(c.params):
                    ftype = c.params[i]
                    break
            fields.append((f"_f{i}", ftype))
        return fields

    def field_index(self, type_name, field_name):
        """Look up a field's index in a struct/sum type. Returns int or None."""
        layout = self._layouts.get(type_name)
        if layout is None:
            return None
        for i, (fn, _ft) in enumerate(layout):
            if fn == field_name:
                return i
        return None

    def layout(self, type_name):
        """Return the field layout for a type."""
        return self._layouts.get(type_name)

    def is_ctor(self, name):
        return name in self.ctors

    def ctor_info(self, name):
        return self.ctors[name]

    def call_output_type(self, proc_name, output_idx):
        """Return the type name for a call output, or None."""
        types = self._proc_output_types.get(proc_name)
        if types is None or output_idx >= len(types):
            return None
        return types[output_idx]


def elaborate(program: LPProgram) -> LPProgram:
    """Elaborate type-level features into core LP Form IR."""
    env = TypeEnv(program)
    check_exhaustive_dispatch(program, env)
    new_procs = [_elaborate_proc(proc, env) for proc in program.procedures]
    return LPProgram(
        procedures=new_procs,
        globals=program.globals,
        arrays=program.arrays,
        structs=program.structs,
        sums=program.sums,
        adts=program.adts,
        entry=program.entry,
    )


def check_exhaustive_dispatch(program: LPProgram, env: "TypeEnv") -> None:
    """Check that procedures dispatching on sum-type constructors are exhaustive.

    For each procedure whose clauses pattern-match on a call output, we
    require that all clauses dispatch on the *same* call in the *same*
    output position, and that the set of constructors they match covers
    every constructor of the sum type exactly once.

    Wildcards (`_`) in dispatch position are rejected: the determinism
    discipline ("exactly one clause fires") is expressed by exhaustive
    non-overlapping constructor coverage, not by a fallback clause.

    Mixing pattern dispatch with plain-variable clauses in the same
    procedure is also rejected.
    """
    errors = []

    for proc in program.procedures:
        if len(proc.clauses) < 2:
            continue

        # For each clause, find the (call_name, output_index, ctor) triple
        # of the first constructor pattern it uses.
        dispatch_sites = []  # one per clause: (call_name, pos, ctor) or None
        saw_wildcard = False
        for clause in proc.clauses:
            site = _first_pattern_site(clause)
            dispatch_sites.append(site)
            if site is not None and site[2] == "_":
                saw_wildcard = True

        any_patterns = any(s is not None for s in dispatch_sites)
        if not any_patterns:
            continue

        if saw_wildcard:
            errors.append(
                f"{proc.name}: wildcard '_' not allowed in dispatch "
                f"position; use exhaustive constructor coverage instead")
            continue

        if any(s is None for s in dispatch_sites):
            errors.append(
                f"{proc.name}: mixes pattern dispatch with plain-variable "
                f"clauses; all clauses must pattern-match")
            continue

        call_names = {s[0] for s in dispatch_sites}
        positions = {s[1] for s in dispatch_sites}
        if len(call_names) != 1 or len(positions) != 1:
            errors.append(
                f"{proc.name}: clauses dispatch on different calls or "
                f"output positions ({sorted(call_names)})")
            continue

        ctors = [s[2] for s in dispatch_sites]
        # Duplicates are allowed — multiple clauses per constructor are
        # disambiguated by further guards on the bound payload variables.

        # Find the sum type that covers these constructors.
        sum_decls = {env.ctor_info(c)[0].name for c in ctors
                     if env.is_ctor(c)}
        unknown = [c for c in ctors if not env.is_ctor(c)]
        if unknown:
            errors.append(
                f"{proc.name}: unknown constructor(s): "
                f"{', '.join(sorted(unknown))}")
            continue
        if len(sum_decls) != 1:
            errors.append(
                f"{proc.name}: clauses dispatch on constructors from "
                f"multiple sum types ({sorted(sum_decls)})")
            continue

        sum_name = next(iter(sum_decls))
        expected = {c.name for c in env.sums[sum_name].constructors}
        missing = expected - set(ctors)
        if missing:
            errors.append(
                f"{proc.name}: non-exhaustive dispatch on {sum_name}; "
                f"missing constructor(s): {', '.join(sorted(missing))}")

    if errors:
        raise ValueError("exhaustiveness errors:\n" +
                         "\n".join(f"  - {e}" for e in errors))


def _first_pattern_site(clause):
    """Return (call_name, output_position, ctor_name) for the first call
    in `clause` whose outputs contain a constructor pattern, else None.
    Non-pattern output positions are skipped; only the pattern one is
    reported. If multiple outputs are patterns, the first is used.
    """
    for goal in clause.goals:
        if isinstance(goal, Call):
            for pos, out in enumerate(goal.outputs):
                if isinstance(out, LPPattern):
                    return (goal.name, pos, out.ctor)
    return None


def _elaborate_proc(proc, env):
    new_clauses = [_elaborate_clause(c, env) for c in proc.clauses]
    return LPProc(
        name=proc.name,
        arity_in=proc.arity_in,
        arity_out=proc.arity_out,
        clauses=new_clauses,
        invariant=proc.invariant,
        measure=proc.measure,
        invertible=proc.invertible,
        output_types=proc.output_types,
    )


def _elaborate_clause(clause, env):
    goals = []
    counter = [0]
    var_types = {}  # variable name -> type name

    def fresh():
        counter[0] += 1
        return f"__elab{counter[0]}"

    def resolve_val(val):
        """Resolve a value, expanding field accesses to struct_get PrimOps."""
        if isinstance(val, LPFieldAccess):
            inner = resolve_val(val.expr)
            if not isinstance(inner, LPVar):
                raise ValueError(
                    f"field access on non-variable: {val.expr}")
            var_name = inner.name
            type_name = var_types.get(var_name)
            if type_name is None:
                raise ValueError(
                    f"cannot resolve field .{val.field}: "
                    f"unknown type for variable '{var_name}'")
            idx = env.field_index(type_name, val.field)
            if idx is None:
                raise ValueError(
                    f"type '{type_name}' has no field '{val.field}'")
            tmp = fresh()
            goals.append(PrimOp(
                "struct_get", [inner], [tmp],
                meta={"type": type_name, "field": val.field, "index": idx},
            ))
            return LPVar(tmp)
        return val

    for goal in clause.goals:
        if isinstance(goal, Guard):
            left = resolve_val(goal.left)
            right = resolve_val(goal.right)
            goals.append(Guard(goal.op, left, right))

        elif isinstance(goal, PrimOp):
            new_inputs = [resolve_val(v) for v in goal.inputs]

            # Track types from struct_new
            if goal.op == "struct_new" and goal.outputs:
                # First input is the type name (as LPVar)
                if new_inputs and isinstance(new_inputs[0], LPVar):
                    type_name = new_inputs[0].name
                    if type_name in env.structs or type_name in env.sums:
                        for out in goal.outputs:
                            var_types[out] = type_name

            goals.append(PrimOp(
                goal.op, new_inputs, goal.outputs, meta=goal.meta))

        elif isinstance(goal, Call):
            new_inputs = [resolve_val(v) for v in goal.inputs]
            new_outputs = []
            destructuring = []  # goals to emit AFTER the call

            for oi, out in enumerate(goal.outputs):
                if isinstance(out, LPPattern):
                    cell_var = fresh()
                    new_outputs.append(cell_var)

                    if out.ctor == "_":
                        continue

                    if env.is_ctor(out.ctor):
                        sum_decl, ctor, tag_idx = env.ctor_info(out.ctor)
                        type_name = sum_decl.name
                        var_types[cell_var] = type_name

                        tag_var = fresh()
                        destructuring.append(PrimOp(
                            "struct_get", [LPVar(cell_var)], [tag_var],
                            meta={"type": type_name, "field": "__tag",
                                  "index": 0},
                        ))
                        destructuring.append(Guard("eq", LPVar(tag_var),
                                                   LPConst(tag_idx)))

                        layout = env.layout(type_name)
                        for i, var_name in enumerate(out.vars):
                            payload_idx = i + 1  # offset past tag
                            if payload_idx < len(layout):
                                field_name = layout[payload_idx][0]
                            else:
                                field_name = f"_f{i}"
                            destructuring.append(PrimOp(
                                "struct_get", [LPVar(cell_var)],
                                [var_name],
                                meta={"type": type_name, "field": field_name,
                                      "index": payload_idx},
                            ))
                    else:
                        raise ValueError(
                            f"unknown constructor '{out.ctor}'")
                else:
                    new_outputs.append(out)
                    # Track type from callee's output_types
                    callee_type = env.call_output_type(goal.name, oi)
                    if callee_type is not None:
                        var_types[out] = callee_type

            # Emit the call first, then destructuring goals
            goals.append(Call(
                goal.name, new_inputs, new_outputs,
                is_tail=goal.is_tail))
            goals.extend(destructuring)

        else:
            goals.append(goal)

    return LPClause(head=clause.head, goals=goals)
