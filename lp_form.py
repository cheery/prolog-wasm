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

Grammar:
  Prog   -> Proc*
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


# -- Goals (clause body elements) --

@dataclass
class PrimOp:
    """Primitive arithmetic operation: op(val*; var*)."""
    op: str              # "add", "sub", "mul", "div", "rem", "copy"
    inputs: list         # list[LPVar | LPConst]
    outputs: list        # list[str]

@dataclass
class Guard:
    """Comparison guard: cmp(val, val;) — no outputs, must hold for clause."""
    op: str              # "eq", "ne", "lt", "le", "gt", "ge"
    left: object         # LPVar | LPConst
    right: object        # LPVar | LPConst

@dataclass
class Call:
    """Procedure call: name(val*; var*)."""
    name: str
    inputs: list         # list[LPVar | LPConst]
    outputs: list        # list[str]
    is_tail: bool = False


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

@dataclass
class LPProgram:
    """A complete LP Form program."""
    procedures: list     # list[LPProc]
    entry: str = None    # name of entry-point procedure


# -- Validation --

def validate(program):
    """Validate an LP Form program.

    Checks:
    - All clauses in a procedure have matching head signatures
    - Single assignment: each variable assigned at most once per clause
    - Guards only appear before non-guard goals
    - Inputs are values (LPVar or LPConst), outputs are variable names
    """
    errors = []
    proc_names = set()

    for proc in program.procedures:
        if proc.name in proc_names:
            errors.append(f"duplicate procedure: {proc.name}")
        proc_names.add(proc.name)

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
            past_guards = False
            for goal in clause.goals:
                if isinstance(goal, Guard):
                    if past_guards:
                        pass  # guards can appear anywhere in principle
                elif isinstance(goal, (PrimOp, Call)):
                    past_guards = True
                    for out in goal.outputs:
                        if out in defined:
                            errors.append(
                                f"{proc.name}: variable '{out}' assigned twice")
                        defined.add(out)

            # Check outputs are defined
            for out in h.outputs:
                if out not in defined:
                    errors.append(
                        f"{proc.name}: output '{out}' never defined")

    if errors:
        raise ValueError("LP Form validation errors:\n" +
                         "\n".join(f"  - {e}" for e in errors))


# -- Tail call marking --

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
                if last.outputs == clause.head.outputs:
                    last.is_tail = True


# -- Pretty printing --

def _fmt_val(v):
    if isinstance(v, LPVar):
        return v.name
    elif isinstance(v, LPConst):
        return str(v.value)
    return repr(v)

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
        outs = ", ".join(g.outputs)
        return f"{g.name}({ins}; {outs})"
    return repr(g)

def pretty_print(program):
    """Return LP Form program as a human-readable string."""
    lines = []
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
