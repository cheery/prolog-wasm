"""LP Form: Horn clause intermediate representation.

Based on the "LP Form" language from:
  Gange, Navas, Schachte, Sondergaard, Stuckey.
  "Horn Clauses as an Intermediate Representation for Program Analysis
   and Transformation" (TPLP 2015).

LP Form is a restricted, deterministic logic programming language:
  - Procedures have multiple clauses with complementary guards
  - Exactly one clause fires per call (no backtracking)
  - Single-moded: inputs are values, outputs are fresh variables
  - Loops expressed as tail recursion
  - Variables scoped to individual clauses

Extended with:
  - Global mutable i32 variables (gget/gset)
  - Mutable i32 arrays (aget/aset/anew)
  - Mutable funcref arrays (rget/rset/rnew) for WAM continuation/BP stacks
  - Multi-clause compound guards
  - Void (zero-output) procedures for side-effecting operations

Grammar:
  Prog   -> GlobalDecl* ArrayDecl* Proc*
  Proc   -> Clause*
  Clause -> Head <- Goal*
  Head   -> Name(Var*; Var*)
  Goal   -> op(Val*; Var*)       # primitive arithmetic
          | cmp(Val, Val;)       # guard (comparison)
          | Name(Val*; Var*)     # procedure call
  Val    -> Var | Const
"""

from dataclasses import dataclass, field


# -- Values --

@dataclass
class LPVar:
    """A variable reference."""
    name: str

@dataclass
class LPConst:
    """An integer constant."""
    value: int

@dataclass
class LPFieldAccess:
    """Field access on a structured value: expr.field_name."""
    expr: object         # LPVar | LPConst | LPFieldAccess
    field: str


# -- Goals (clause body elements) --

@dataclass
class PrimOp:
    """Primitive arithmetic/state operation: op(val*; var*)."""
    op: str              # "add", "sub", "mul", "div", "rem", "copy",
                         # "gget", "gset", "aget", "aset", "anew",
                         # "rget", "rset", "rnew",
                         # "struct_new", "struct_get"
    inputs: list         # list[LPVar | LPConst]
    outputs: list        # list[str]  (may be empty for side-effecting ops)
    meta: dict = None    # optional: {"type": "Cell", "field": "tag"} etc.

@dataclass
class Guard:
    """Comparison guard: cmp(val, val;) — no outputs, must hold for clause."""
    op: str              # "eq", "ne", "lt", "le", "gt", "ge"
    left: object         # LPVar | LPConst | LPFieldAccess
    right: object        # LPVar | LPConst | LPFieldAccess

@dataclass
class Call:
    """Procedure call: name(val*; var*)."""
    name: str
    inputs: list         # list[LPVar | LPConst]
    outputs: list        # list[str | LPPattern]
    is_tail: bool = False


# -- Patterns (constructor patterns in call output positions) --

@dataclass
class LPPattern:
    """Constructor pattern in a call output: ctor(var1, var2, ...) or _."""
    ctor: str            # constructor name, or "_" for wildcard
    vars: list           # list[str] — bound variable names


# -- Type Declarations --

@dataclass
class LPStructDecl:
    """A struct type: struct Name { field: Type, ... }"""
    name: str
    fields: list         # list[tuple[str, str]] — (field_name, field_type)

@dataclass
class LPConstructor:
    """A constructor of a sum type: Ctor(Type, Type, ...)"""
    name: str
    params: list         # list[str] — parameter type names

@dataclass
class LPSumDecl:
    """A sum type: type Name = Ctor1(Types) | Ctor2(Types) | ..."""
    name: str
    constructors: list   # list[LPConstructor]


@dataclass
class LPSignature:
    """An ADT operation signature: name(arity_in; arity_out)."""
    name: str
    arity_in: int
    arity_out: int

@dataclass
class LPADT:
    """An abstract data type: adt Name { sig1; sig2; ... }.

    An ADT is a logical interface: each signature names an operation of
    the ADT with an input/output arity. The procedures implementing the
    ADT live in the program's regular procedure list; the ADT block is
    validated against those procedures' actual arities.

    Rationale: ADTs exist to (a) document the interface boundary for
    modular verification, and (b) eventually carry compiler hints for
    the dual WASM/CHC projection. For now the
    block is descriptive — the emitter and CHC extractor use the
    procedures directly.
    """
    name: str
    signatures: list     # list[LPSignature]


# -- State Declarations --

@dataclass
class GlobalDecl:
    """Global mutable i32 variable."""
    name: str
    initial: int = 0

@dataclass
class ArrayDecl:
    """Named mutable array. kind is "i32" or "ref"."""
    name: str
    kind: str = "i32"    # "i32" for i32 arrays, "ref" for funcref arrays


# -- Structure --

@dataclass
class LPHead:
    """Clause head: name(inputs; outputs)."""
    name: str
    inputs: list         # list[str] — input parameter names
    outputs: list        # list[str] — output parameter names

@dataclass
class LPClause:
    """A single clause: head <- goal1, goal2, ..."""
    head: LPHead
    goals: list          # list[PrimOp | Guard | Call]

@dataclass
class LPProc:
    """A procedure: set of clauses with the same head signature."""
    name: str
    arity_in: int
    arity_out: int
    clauses: list        # list[LPClause]
    invariant: object = None  # future: CHC-inferred invariant
    measure: object = None    # termination measure: list[str] of input names
    invertible: bool = False  # if True, emitter skips trace writes for this
                              # proc (Phase 7e). Must be a pure leaf proc —
                              # no Calls, no gset/aset/anew/rnew.
    output_types: list = None  # list of output type names (None for i32).
                               # Computed by infer_output_types().

@dataclass
class LPProgram:
    """A complete LP Form program."""
    procedures: list     # list[LPProc]
    globals: list = field(default_factory=list)   # list[GlobalDecl]
    arrays: list = field(default_factory=list)    # list[ArrayDecl]
    structs: list = field(default_factory=list)   # list[LPStructDecl]
    sums: list = field(default_factory=list)      # list[LPSumDecl]
    adts: list = field(default_factory=list)      # list[LPADT]
    entry: str = None    # name of entry-point procedure

    def link(self, other: "LPProgram") -> "LPProgram":
        """Merge another LPProgram into this one.

        Concatenates procedures, globals, and arrays. Duplicate global
        declarations (same name AND same initial value) and duplicate
        array declarations (same name AND same kind) are deduplicated
        silently; mismatches and procedure name clashes raise.

        The result uses self.entry. The other program's entry is ignored.
        """
        my_procs = {p.name for p in self.procedures}
        for p in other.procedures:
            if p.name in my_procs:
                raise ValueError(
                    f"link: duplicate procedure '{p.name}'")

        my_globals = {g.name: g for g in self.globals}
        merged_globals = list(self.globals)
        for g in other.globals:
            if g.name in my_globals:
                existing = my_globals[g.name]
                if existing.initial != g.initial:
                    raise ValueError(
                        f"link: global '{g.name}' initial value mismatch "
                        f"({existing.initial} vs {g.initial})")
            else:
                merged_globals.append(g)
                my_globals[g.name] = g

        my_arrays = {a.name: a for a in self.arrays}
        merged_arrays = list(self.arrays)
        for a in other.arrays:
            if a.name in my_arrays:
                existing = my_arrays[a.name]
                if existing.kind != a.kind:
                    raise ValueError(
                        f"link: array '{a.name}' kind mismatch "
                        f"({existing.kind} vs {a.kind})")
            else:
                merged_arrays.append(a)
                my_arrays[a.name] = a

        my_structs = {s.name: s for s in self.structs}
        merged_structs = list(self.structs)
        for s in other.structs:
            if s.name in my_structs:
                raise ValueError(
                    f"link: duplicate struct '{s.name}'")
            merged_structs.append(s)

        my_sums = {s.name: s for s in self.sums}
        merged_sums = list(self.sums)
        for s in other.sums:
            if s.name in my_sums:
                raise ValueError(
                    f"link: duplicate sum type '{s.name}'")
            merged_sums.append(s)

        my_adts = {a.name: a for a in self.adts}
        merged_adts = list(self.adts)
        for a in other.adts:
            if a.name in my_adts:
                raise ValueError(
                    f"link: duplicate adt '{a.name}'")
            merged_adts.append(a)

        return LPProgram(
            procedures=list(self.procedures) + list(other.procedures),
            globals=merged_globals,
            arrays=merged_arrays,
            structs=merged_structs,
            sums=merged_sums,
            adts=merged_adts,
            entry=self.entry,
        )


# -- Validation --

def validate(program):
    """Validate an LP Form program.

    Checks:
    - All clauses in a procedure have matching head signatures
    - Single assignment: each variable assigned at most once per clause
    - Inputs are values (LPVar or LPConst), outputs are variable names
    """
    errors = []
    proc_names = set()

    for proc in program.procedures:
        if proc.name in proc_names:
            errors.append(f"duplicate procedure: {proc.name}")
        proc_names.add(proc.name)

        # Collect declared global/array names for skipping output checks
        declared_names = set()
        for g in program.globals:
            declared_names.add(g.name)
        for a in program.arrays:
            declared_names.add(a.name)

        for clause in proc.clauses:
            h = clause.head
            if h.name != proc.name:
                errors.append(f"clause head {h.name} != proc {proc.name}")
            if len(h.inputs) != proc.arity_in:
                errors.append(f"{proc.name}: input arity mismatch")
            if len(h.outputs) != proc.arity_out:
                errors.append(f"{proc.name}: output arity mismatch")

            # Check single assignment
            defined = set(h.inputs)  # inputs are defined on entry
            for goal in clause.goals:
                if isinstance(goal, (PrimOp, Call)):
                    for out in goal.outputs:
                        if isinstance(out, LPPattern):
                            for v in out.vars:
                                if v in defined:
                                    errors.append(
                                        f"{proc.name}: variable '{v}' "
                                        f"assigned twice")
                                defined.add(v)
                        else:
                            if out in defined:
                                errors.append(
                                    f"{proc.name}: variable '{out}' "
                                    f"assigned twice")
                            defined.add(out)

            # Check outputs are defined (skip for void procedures)
            for out in h.outputs:
                if out not in defined:
                    errors.append(
                        f"{proc.name}: output '{out}' never defined")

    # ADT signatures must match implementing procedures.
    proc_sigs = {p.name: (p.arity_in, p.arity_out)
                 for p in program.procedures}
    for adt in getattr(program, "adts", []) or []:
        for sig in adt.signatures:
            actual = proc_sigs.get(sig.name)
            if actual is None:
                errors.append(
                    f"adt {adt.name}: signature '{sig.name}' has no "
                    f"implementing procedure")
                continue
            if actual != (sig.arity_in, sig.arity_out):
                errors.append(
                    f"adt {adt.name}: signature '{sig.name}' has arity "
                    f"({sig.arity_in};{sig.arity_out}) but procedure has "
                    f"({actual[0]};{actual[1]})")

    if errors:
        raise ValueError("LP Form validation errors:\n" +
                         "\n".join(f"  - {e}" for e in errors))


# -- Tail call marking --

def _flatten_outputs(outputs):
    """Flatten a list of outputs that may contain LPPattern into plain names."""
    result = []
    for o in outputs:
        if isinstance(o, LPPattern):
            result.extend(o.vars)
        else:
            result.append(o)
    return result


def mark_tail_calls(program):
    """Mark calls that are in tail position.

    A Call is tail if it's the last goal in the clause and its outputs
    match the clause head's outputs exactly.
    """
    for proc in program.procedures:
        for clause in proc.clauses:
            if not clause.goals:
                continue
            last = clause.goals[-1]
            if isinstance(last, Call):
                flat = _flatten_outputs(last.outputs)
                if flat == clause.head.outputs:
                    last.is_tail = True


# -- Pretty printing --

def _fmt_val(v):
    if isinstance(v, LPVar):
        return v.name
    elif isinstance(v, LPConst):
        return str(v.value)
    elif isinstance(v, LPFieldAccess):
        return f"{_fmt_val(v.expr)}.{v.field}"
    return repr(v)

def _fmt_out(o):
    if isinstance(o, LPPattern):
        if o.ctor == "_":
            return "_"
        if not o.vars:
            return o.ctor
        return f"{o.ctor}({', '.join(o.vars)})"
    return str(o)

def _fmt_goal(g):
    if isinstance(g, Guard):
        op_sym = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=",
                  "gt": ">", "ge": ">="}.get(g.op, g.op)
        return f"{_fmt_val(g.left)} {op_sym} {_fmt_val(g.right)}"
    elif isinstance(g, PrimOp):
        ins = ", ".join(_fmt_val(v) for v in g.inputs)
        outs = ", ".join(g.outputs)
        return f"{g.op}({ins}; {outs})"
    elif isinstance(g, Call):
        ins = ", ".join(_fmt_val(v) for v in g.inputs)
        outs = ", ".join(_fmt_out(o) for o in g.outputs)
        return f"{g.name}({ins}; {outs})"
    return repr(g)

def pretty_print(program):
    """Return LP Form program as a human-readable string."""
    lines = []
    for g in program.globals:
        lines.append(f"global {g.name} = {g.initial}.")
    for a in program.arrays:
        lines.append(f"array {a.name}.")
    for s in program.structs:
        fields = ", ".join(f"{fn}: {ft}" for fn, ft in s.fields)
        lines.append(f"struct {s.name} {{ {fields} }}.")
    for s in program.sums:
        ctors = " | ".join(
            f"{c.name}({', '.join(c.params)})" if c.params else c.name
            for c in s.constructors)
        lines.append(f"type {s.name} = {ctors}.")
    for a in (program.adts or []):
        sig_lines = []
        for sig in a.signatures:
            ins = ", ".join(f"_i{i}" for i in range(sig.arity_in))
            outs = ", ".join(f"_o{i}" for i in range(sig.arity_out))
            sig_lines.append(f"    {sig.name}({ins}; {outs})")
        lines.append(f"adt {a.name} {{")
        lines.extend(sig_lines)
        lines.append("}.")
    if (program.globals or program.arrays or program.structs or
            program.sums or program.adts):
        lines.append("")
    for proc in program.procedures:
        for clause in proc.clauses:
            h = clause.head
            ins = ", ".join(h.inputs)
            outs = ", ".join(h.outputs)
            head_str = f"{h.name}({ins}; {outs})"
            if clause.goals:
                body = " /\\ ".join(_fmt_goal(g) for g in clause.goals)
                lines.append(f"{head_str} <- {body}")
            else:
                lines.append(head_str)
        lines.append("")
    return "\n".join(lines)


# -- Variable collection helpers --

def collect_vars_from_val(v, vs):
    """Add variable names referenced by a value to set `vs`."""
    if isinstance(v, LPVar):
        vs.add(v.name)
    elif isinstance(v, LPFieldAccess):
        collect_vars_from_val(v.expr, vs)


def flatten_outputs(outputs):
    """Flatten outputs list: LPPattern -> individual var names."""
    result = []
    for o in outputs:
        if isinstance(o, LPPattern):
            result.extend(o.vars)
        else:
            result.append(o)
    return result


def infer_output_types(program):
    """Infer the WASM result types for each procedure's outputs.

    Populates proc.output_types as a list of type names (str) or None (i32).
    A proc's output is a ref type if any clause assigns a struct_new result
    to the corresponding head output variable.

    Returns the program (mutated in place).
    """
    struct_names = {s.name for s in program.structs}
    sum_names = {s.name for s in program.sums}
    all_type_names = struct_names | sum_names
    proc_by_name = {p.name: p for p in program.procedures}

    # Phase 1: direct struct_new assignments
    for proc in program.procedures:
        types = [None] * proc.arity_out
        for clause in proc.clauses:
            # Map: var_name -> output_position
            out_vars = {}
            for i, o in enumerate(clause.head.outputs):
                out_vars[o] = i

            for goal in clause.goals:
                if isinstance(goal, PrimOp) and goal.op == "struct_new":
                    type_name = goal.inputs[0].name if isinstance(goal.inputs[0], LPVar) else None
                    if type_name in all_type_names:
                        for out in goal.outputs:
                            pos = out_vars.get(out)
                            if pos is not None:
                                types[pos] = type_name
                if isinstance(goal, PrimOp) and goal.op == "copy":
                    # copy doesn't change the type — but if the source is
                    # from a call we haven't typed yet, we'll catch it in
                    # phase 2.
                    pass
                if isinstance(goal, Call):
                    callee = proc_by_name.get(goal.name)
                    if callee is not None and callee.output_types is not None:
                        for i, out in enumerate(goal.outputs):
                            if isinstance(out, str) and i < len(callee.output_types):
                                ct = callee.output_types[i]
                                if ct is not None:
                                    pos = out_vars.get(out)
                                    if pos is not None and types[pos] is None:
                                        types[pos] = ct

        proc.output_types = types

    # Phase 2: propagate call results (fixed point)
    changed = True
    while changed:
        changed = False
        for proc in program.procedures:
            for clause in proc.clauses:
                out_vars = {}
                for i, o in enumerate(clause.head.outputs):
                    out_vars[o] = i

                for goal in clause.goals:
                    if isinstance(goal, Call):
                        callee = proc_by_name.get(goal.name)
                        if callee is None or callee.output_types is None:
                            continue
                        for i, out in enumerate(goal.outputs):
                            if isinstance(out, str) and i < len(callee.output_types):
                                ct = callee.output_types[i]
                                if ct is not None:
                                    pos = out_vars.get(out)
                                    if pos is not None and proc.output_types[pos] is None:
                                        proc.output_types[pos] = ct
                                        changed = True

    return program
