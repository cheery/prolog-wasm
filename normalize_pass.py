"""NormalizeLists pass: L0 -> L1.

Desugars three syntactic-sugar forms that prolog_parser.py produces but
the WAM compiler does not need to handle:

  List(heads, tail)           ->  Compound('.', [head, Compound('.', ...)])
                                  or Atom('[]') for the empty list
  BinOp(op, left, right)      ->  Compound(op, [left, right])
  UnaryOp(op, operand)        ->  Compound(op, [operand])

All other node types (Atom, Number, Var, Compound, Fact, Rule, Query,
Program) are copied unchanged via the pass expander's auto-generated
identity visitors.
"""

from nanopass import Pass
from languages import L0, L1
from prolog_parser import Atom, Compound


class NormalizeLists(Pass, source=L0, target=L1):
    """Desugar L0 surface syntax into the simpler L1 term language."""

    def visit_List(self, node):
        # Build a right-nested ./2 chain.
        # Tail: if given, normalize it; otherwise use the '[]' atom.
        if node.tail is not None:
            tail = self.visit(node.tail)
        else:
            tail = Atom('[]')
        for head in reversed(node.heads):
            tail = Compound('.', [self.visit(head), tail])
        return tail

    def visit_BinOp(self, node):
        # BinOp(op, left, right) -> Compound(op, [left, right])
        return Compound(node.op, [self.visit(node.left), self.visit(node.right)])

    def visit_UnaryOp(self, node):
        # UnaryOp(op, operand) -> Compound(op, [operand])
        return Compound(node.op, [self.visit(node.operand)])
