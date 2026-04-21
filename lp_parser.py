"""LP Form parser using Lark.

Parses a small language based on the LP Form from Gange et al. 2015.

Syntax:

    // Declarations
    global H = 0.
    global B = -1.
    array HEAP.
    array CONT: ref.

    // Procedures
    gcd(a, b; ret): b != 0, mod(a, b; b'), gcd(b, b'; ret).
    gcd(a, b; a): b == 0.

    // Void procedures (no outputs)
    heap_set_tag(addr, tag;):
        mul(addr, 2; idx), aset(HEAP, idx, tag;).

Rules:
  - Semicolon separates inputs from outputs in heads and goals.
  - Colon introduces the body; dot ends the clause.
  - Variables are identifiers, optionally primed (b', n').
  - Numbers are integer literals.
  - Guards are inline comparisons: ==, !=, <, <=, >, >=
  - If a head output is an input variable, it means implicit copy.
  - A clause with no body (just head and dot) is a unit clause.
  - Built-in PrimOps: add, sub, mul, div, rem/mod, copy, neg,
    gget, gset, aget, aset, anew, rnew.
"""

from lark import Lark, Transformer, v_args
from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst, LPFieldAccess, LPPattern,
    LPStructDecl, LPConstructor, LPSumDecl,
    LPADT, LPSignature,
    GlobalDecl, ArrayDecl,
    validate, mark_tail_calls,
)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

LP_GRAMMAR = r"""
    start: decl* clause+

    // Declarations
    ?decl: global_decl | array_decl | struct_decl | type_decl | adt_decl

    global_decl: "global" NAME "=" SIGNED_NUMBER "."
    array_decl: "array" NAME "."
              | "array" NAME ":" NAME "."

    // Type declarations
    struct_decl: "struct" NAME "{" field_list "}" "."
    type_decl: "type" NAME "=" ctor_list "."

    // Abstract data type declarations
    adt_decl: "adt" NAME "{" sig_list "}" "."

    sig_list: sig*
    sig: NAME "(" in_params ";" sig_out_params ")"
       | NAME "(" ";" sig_out_params ")"
       | NAME "(" in_params ";" ")"
       | NAME "(" ";" ")"

    sig_out_params: NAME ("," NAME)*

    field_list: field ("," field)*
    field: NAME ":" NAME

    ctor_list: ctor ("|" ctor)*
    ctor: NAME "(" type_list ")" | NAME

    type_list: NAME ("," NAME)*

    // Clauses
    clause: head ":" body "."   -> clause_with_body
          | head "."            -> clause_no_body

    head: NAME "(" in_params ";" out_vals ")"
        | NAME "(" ";" out_vals ")"
        | NAME "(" in_params ";" ")"
        | NAME "(" ";" ")"

    in_params: NAME ("," NAME)*

    out_vals: val ("," val)*

    body: goal ("," goal)*

    ?goal: guard | call_goal

    guard: val CMP val

    call_goal: NAME "(" args_io ")"
             | NAME "(" ")"
             | NAME "(" ";" ")"

    args_io: vals ";" out_names    -> call_with_outputs
           | vals ";"              -> call_no_outputs
           | ";" out_names         -> call_no_inputs
           | vals                  -> call_inputs_only

    vals: val ("," val)*

    // Output names: plain variables or constructor patterns
    out_names: out_name ("," out_name)*

    ?out_name: NAME "(" out_var_list ")" -> ctor_out_pattern
             | NAME                      -> plain_out_name

    out_var_list: NAME ("," NAME)* | -> empty_vars

    ?val: NAME  -> var_val
        | SIGNED_NUMBER -> num_val
        | NAME "." NAME -> field_access

    CMP: "!=" | "==" | "<=" | ">=" | "<" | ">"

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*'*/

    SIGNED_NUMBER: /\-?[0-9]+/

    %import common.WS
    %ignore WS

    COMMENT: /\/\/.*/
    %ignore COMMENT

    BLOCK_COMMENT: "/*" /[\s\S]*?/ "*/"
    %ignore BLOCK_COMMENT
"""


# ---------------------------------------------------------------------------
# Transformer: parse tree -> LP Form IR
# ---------------------------------------------------------------------------

# Map syntax operators to Guard op names
_CMP_MAP = {
    "==": "eq",
    "!=": "ne",
    "<":  "lt",
    "<=": "le",
    ">":  "gt",
    ">=": "ge",
}

# Built-in primitive operations (not procedure calls)
_BUILTINS = {
    "add", "sub", "mul", "div", "rem", "mod", "copy", "neg",
    "gget", "gset", "aget", "aset", "anew", "rnew",
    "and", "or",
    "struct_new", "struct_get",
}


@v_args(inline=True)
class LPTransformer(Transformer):

    def start(self, *items):
        return list(items)

    def global_decl(self, name, value):
        return GlobalDecl(name=str(name), initial=int(str(value)))

    def array_decl(self, name, kind=None):
        k = str(kind) if kind is not None else "i32"
        return ArrayDecl(name=str(name), kind=k)

    def struct_decl(self, name, fields):
        field_list = [(str(fn), str(ft)) for fn, ft in fields]
        return LPStructDecl(name=str(name), fields=field_list)

    def type_decl(self, name, ctors):
        return LPSumDecl(name=str(name), constructors=list(ctors))

    def adt_decl(self, name, sigs):
        return LPADT(name=str(name), signatures=list(sigs))

    def sig_list(self, *sigs):
        return list(sigs)

    def sig(self, name, *rest):
        name = str(name)
        arity_in = 0
        arity_out = 0
        for item in rest:
            if isinstance(item, list) and item and isinstance(item[0], str) \
               and item[0] == "__inputs__":
                arity_in = len(item) - 1
            elif isinstance(item, list) and item and isinstance(item[0], str) \
                 and item[0] == "__sig_outs__":
                arity_out = len(item) - 1
        return LPSignature(name=name, arity_in=arity_in, arity_out=arity_out)

    def sig_out_params(self, *names):
        return ["__sig_outs__"] + [str(n) for n in names]

    def field_list(self, *fields):
        return list(fields)

    def field(self, name, typ):
        return (name, typ)

    def ctor_list(self, *ctors):
        return list(ctors)

    def ctor(self, name, *rest):
        params = []
        for r in rest:
            if isinstance(r, list):
                params = [str(t) for t in r]
        return LPConstructor(name=str(name), params=params)

    def type_list(self, *names):
        return list(names)

    def empty_vars(self, ):
        return []

    def clause_with_body(self, head, body):
        head_info, output_copies = head
        goals = body + output_copies
        return LPClause(head=head_info, goals=goals)

    def clause_no_body(self, head):
        head_info, output_copies = head
        return LPClause(head=head_info, goals=output_copies)

    def head(self, name, *rest):
        name = str(name)
        inputs = []
        out_vals = []
        for item in rest:
            if isinstance(item, list) and item and isinstance(item[0], str) \
               and item[0] == "__inputs__":
                inputs = item[1:]
            elif isinstance(item, list) and item and isinstance(item[0], str) \
               and item[0] == "__outvals__":
                out_vals = item[1:]

        output_names = []
        copy_goals = []
        for i, v in enumerate(out_vals):
            if isinstance(v, LPVar) and v.name not in inputs:
                output_names.append(v.name)
            elif isinstance(v, LPVar) and v.name in inputs:
                out_name = f"_ret{i}"
                output_names.append(out_name)
                copy_goals.append(
                    PrimOp("copy", [LPVar(v.name)], [out_name]))
            elif isinstance(v, LPConst):
                out_name = f"_ret{i}"
                output_names.append(out_name)
                copy_goals.append(
                    PrimOp("copy", [v], [out_name]))
            else:
                output_names.append(str(v))

        head_info = LPHead(name=name, inputs=inputs, outputs=output_names)
        return (head_info, copy_goals)

    def in_params(self, *names):
        return ["__inputs__"] + [str(n) for n in names]

    def out_vals(self, *vals):
        return ["__outvals__"] + list(vals)

    def body(self, *goals):
        return list(goals)

    def guard(self, left, op, right):
        return Guard(op=_CMP_MAP[str(op).strip()], left=left, right=right)

    def call_goal(self, name, *rest):
        name = str(name)
        inputs = []
        outputs = []
        for item in rest:
            if isinstance(item, tuple):
                inputs, outputs = item

        if name in _BUILTINS:
            actual_op = name
            if actual_op == "mod":
                actual_op = "rem"
            return PrimOp(op=actual_op, inputs=inputs, outputs=outputs)
        else:
            return Call(name=name, inputs=inputs, outputs=outputs)

    def call_with_outputs(self, vals, names):
        return (vals, names)

    def call_no_outputs(self, vals):
        return (vals, [])

    def call_no_inputs(self, names):
        return ([], names)

    def call_inputs_only(self, vals):
        return (vals, [])

    def vals(self, *vs):
        return list(vs)

    def names(self, *ns):
        return [str(n) for n in ns]

    def out_names(self, *ns):
        return list(ns)

    def plain_out_name(self, token):
        return str(token)

    def ctor_out_pattern(self, ctor, *rest):
        vars_ = []
        for r in rest:
            if isinstance(r, list):
                vars_ = [str(v) for v in r]
        return LPPattern(ctor=str(ctor), vars=vars_)

    def out_var_list(self, *ns):
        return [str(n) for n in ns]

    def var_val(self, token):
        return LPVar(str(token))

    def num_val(self, token):
        return LPConst(int(str(token)))

    def field_access(self, obj, field):
        return LPFieldAccess(LPVar(str(obj)), str(field))


# ---------------------------------------------------------------------------
# Grouping clauses into procedures
# ---------------------------------------------------------------------------

def _group_clauses(clauses):
    """Group parsed clauses into LPProc instances by head name."""
    procs = {}
    proc_order = []
    for clause in clauses:
        name = clause.head.name
        if name not in procs:
            procs[name] = []
            proc_order.append(name)
        procs[name].append(clause)

    result = []
    for name in proc_order:
        cls = procs[name]
        arity_in = len(cls[0].head.inputs)
        arity_out = len(cls[0].head.outputs)
        result.append(LPProc(
            name=name,
            arity_in=arity_in,
            arity_out=arity_out,
            clauses=cls,
        ))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_parser = Lark(LP_GRAMMAR, parser="earley", ambiguity="resolve")
_transformer = LPTransformer()


def parse_lp(source: str, entry: str = None) -> LPProgram:
    """Parse LP Form source code and return an LPProgram.

    If entry is None, the first procedure is used as the entry point.
    """
    tree = _parser.parse(source)
    items = _transformer.transform(tree)

    globals_ = [i for i in items if isinstance(i, GlobalDecl)]
    arrays = [i for i in items if isinstance(i, ArrayDecl)]
    structs = [i for i in items if isinstance(i, LPStructDecl)]
    sums = [i for i in items if isinstance(i, LPSumDecl)]
    adts = [i for i in items if isinstance(i, LPADT)]
    clauses = [i for i in items if isinstance(i, LPClause)]
    procs = _group_clauses(clauses)

    if entry is None and procs:
        entry = procs[0].name

    return LPProgram(
        procedures=procs,
        globals=globals_,
        arrays=arrays,
        structs=structs,
        sums=sums,
        adts=adts,
        entry=entry,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from lp_form import pretty_print

    source = """
        // GCD — the classic example
        gcd(a, b; ret): b != 0, mod(a, b; b'), gcd(b, b'; ret).
        gcd(a, b; a): b == 0.
    """
    prog = parse_lp(source)
    print(pretty_print(prog))
    print("---")

    source2 = """
        // WAM-style heap operations
        global H = 0.
        array HEAP.

        heap_push(tag, val; old_h):
            gget(H; old_h),
            mul(old_h, 2; idx),
            aset(HEAP, idx, tag;),
            add(idx, 1; idx1),
            aset(HEAP, idx1, val;),
            add(old_h, 1; new_h),
            gset(H, new_h;).

        heap_get_tag(addr; tag):
            mul(addr, 2; idx), aget(HEAP, idx; tag).
    """
    prog2 = parse_lp(source2)
    print(pretty_print(prog2))
