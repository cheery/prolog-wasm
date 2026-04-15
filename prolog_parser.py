"""Prolog parser using Lark.

Parses a subset of Prolog sufficient for a proof-of-concept compiler:
  - Facts:    parent(tom, bob).
  - Rules:    grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
  - Queries:  ?- grandparent(tom, X).
  - Terms:    atoms, numbers, variables, compound terms, lists, strings
  - Operators: arithmetic (+, -, *, /), comparison (=, \\=, <, >, =<, >=, =:=, =\\=),
               unification (=, \\=), is/2, not/1, \\+
"""

from lark import Lark, Transformer, v_args
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    name: str
    def __repr__(self):
        return self.name

@dataclass
class Number:
    value: int | float
    def __repr__(self):
        return str(self.value)

@dataclass
class Var:
    name: str
    def __repr__(self):
        return self.name

@dataclass
class Compound:
    functor: str
    args: list
    def __repr__(self):
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.functor}({args_str})"

@dataclass
class List:
    heads: list
    tail: Any = None  # None means [], otherwise a Var or another list
    def __repr__(self):
        elems = ", ".join(repr(h) for h in self.heads)
        if self.tail is None:
            return f"[{elems}]"
        return f"[{elems} | {repr(self.tail)}]"

@dataclass
class Fact:
    head: Any
    def __repr__(self):
        return f"{repr(self.head)}."

@dataclass
class Rule:
    head: Any
    body: list  # list of goals
    def __repr__(self):
        body_str = ", ".join(repr(g) for g in self.body)
        return f"{repr(self.head)} :- {body_str}."

@dataclass
class Query:
    goals: list
    def __repr__(self):
        return "?- " + ", ".join(repr(g) for g in self.goals) + "."

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    def __repr__(self):
        return f"({repr(self.left)} {self.op} {repr(self.right)})"

@dataclass
class UnaryOp:
    op: str
    operand: Any
    def __repr__(self):
        return f"{self.op}({repr(self.operand)})"

@dataclass
class Program:
    clauses: list  # list of Fact, Rule, Query


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

PROLOG_GRAMMAR = r"""
    start: clause*

    clause: head ":-" body "."  -> rule
          | "?-" body "."       -> query
          | head "."            -> fact

    head: term

    body: goal ("," goal)*
    goal: term

    // Terms with operator precedence (low to high)
    ?term: op_700

    // 700: comparison and unification
    ?op_700: op_500 COMPARE_OP op_500  -> binop
           | op_500 "is" op_500        -> is_op
           | op_500

    COMPARE_OP: "=:=" | "=\\=" | "=<" | ">=" | "\\=" | "==" | "\\=="
              | "=" | "<" | ">"

    // 500: addition, subtraction
    ?op_500: op_500 ADD_OP op_400  -> binop
           | op_400
    ADD_OP: "+" | "-"

    // 400: multiplication, division
    ?op_400: op_400 MUL_OP unary   -> binop
           | unary
    MUL_OP: "*" | "/" | "mod" | "rem"

    // unary minus, \+
    ?unary: "-" primary            -> neg
          | "\\+" primary          -> naf
          | "not" primary          -> naf
          | primary

    ?primary: atom "(" args ")"    -> compound
            | atom                 -> atom_term
            | VARIABLE             -> variable
            | NUMBER               -> number
            | FLOAT                -> float_num
            | string
            | list
            | "(" term ")"
            | "!" -> cut

    args: term ("," term)*

    list: "[" "]"                       -> empty_list
        | "[" list_items "]"            -> list_items_only
        | "[" list_items "|" term "]"   -> list_with_tail

    list_items: term ("," term)*

    atom: ATOM | QUOTED_ATOM

    string: ESCAPED_STRING

    // Terminals
    ATOM: /[a-z][a-zA-Z0-9_]*/
    QUOTED_ATOM: "'" /[^']*/ "'"
    VARIABLE: /[A-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+/
    FLOAT: /[0-9]+\.[0-9]+/

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS

    // Line comments
    COMMENT: /%.*/
    %ignore COMMENT

    // Block comments
    BLOCK_COMMENT: "/*" /[\s\S]*?/ "*/"
    %ignore BLOCK_COMMENT
"""


# ---------------------------------------------------------------------------
# Tree -> AST transformer
# ---------------------------------------------------------------------------

@v_args(inline=True)
class PrologTransformer(Transformer):
    def start(self, *clauses):
        return Program(list(clauses))

    def rule(self, head, body):
        return Rule(head, body)

    def query(self, body):
        return Query(body)

    def fact(self, head):
        return Fact(head)

    def head(self, term):
        return term

    def body(self, *goals):
        return list(goals)

    def goal(self, term):
        return term

    def compound(self, atom, args):
        return Compound(atom.name, args)

    def atom_term(self, atom):
        return atom

    def atom(self, token):
        s = str(token)
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        return Atom(s)

    def variable(self, token):
        name = str(token)
        if name == "_":
            return Var("_")
        return Var(name)

    def number(self, token):
        return Number(int(str(token)))

    def float_num(self, token):
        return Number(float(str(token)))

    def string(self, token):
        # Strings become lists of character codes (Prolog convention)
        s = str(token)[1:-1]  # strip quotes
        return List([Number(ord(c)) for c in s])

    def args(self, *terms):
        return list(terms)

    def empty_list(self):
        return List([])

    def list_items_only(self, items):
        return List(items)

    def list_with_tail(self, items, tail):
        return List(items, tail)

    def list_items(self, *terms):
        return list(terms)

    def binop(self, left, op, right):
        return BinOp(str(op).strip(), left, right)

    def is_op(self, left, right):
        return BinOp("is", left, right)

    def neg(self, operand):
        if isinstance(operand, Number):
            return Number(-operand.value)
        return UnaryOp("-", operand)

    def naf(self, operand):
        return UnaryOp("\\+", operand)

    def cut(self):
        return Atom("!")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_parser = Lark(PROLOG_GRAMMAR, parser="earley", ambiguity="resolve")
_transformer = PrologTransformer()

def parse(source: str) -> Program:
    """Parse Prolog source code and return an AST."""
    tree = _parser.parse(source)
    return _transformer.transform(tree)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_source = """
        % Family relations
        parent(tom, bob).
        parent(tom, liz).
        parent(bob, ann).
        parent(bob, pat).

        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

        ancestor(X, Y) :- parent(X, Y).
        ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

        /* List operations */
        member(X, [X | _]).
        member(X, [_ | T]) :- member(X, T).

        append([], L, L).
        append([H | T], L, [H | R]) :- append(T, L, R).

        length([], 0).
        length([_ | T], N) :- length(T, N1), N is N1 + 1.

        factorial(0, 1).
        factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N1 * F1.

        ?- grandparent(tom, X).
        ?- member(2, [1, 2, 3]).
        ?- append([1, 2], [3, 4], X).
    """

    program = parse(test_source)
    for clause in program.clauses:
        print(clause)
