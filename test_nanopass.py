"""Self-contained tests for the nanopass framework.

These tests use toy languages and do not touch the Prolog compiler at all.
"""

import sys
import dataclasses
from nanopass import Language, Pass, ValidationError, DispatchError


# ---------------------------------------------------------------------------
# Toy language definitions
# ---------------------------------------------------------------------------

class LA(Language):
    """Simple expression language with booleans and if-expressions."""

    class Expr:
        # abstract nonterminal marker; in practice we dispatch by node type
        pass

    class BoolLit:
        value: bool

    class IntLit:
        value: int

    class Add:
        left: object  # Expr
        right: object

    class If:
        cond: object
        then: object
        else_: object

    class Seq:
        first: object
        second: object

    class ListOf:
        items: list


class LB(Language, extends=LA):
    """LA minus BoolLit; adds Not."""
    remove = ['BoolLit']

    class Not:
        operand: object


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ok(label):
    print(f"  PASS  {label}")


def fail(label, msg):
    print(f"  FAIL  {label}: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1: Language node registration
# ---------------------------------------------------------------------------

def test_language_registration():
    assert 'BoolLit' in LA._nodes, "LA should have BoolLit"
    assert 'IntLit' in LA._nodes
    assert 'Add' in LA._nodes
    assert 'If' in LA._nodes
    assert 'Seq' in LA._nodes
    assert 'ListOf' in LA._nodes

    assert 'BoolLit' not in LB._nodes, "LB should not have BoolLit"
    assert 'Not' in LB._nodes, "LB should have Not"
    assert 'IntLit' in LB._nodes, "LB should inherit IntLit"

    # Shared nodes are the exact same Python class
    assert LA._nodes['IntLit'] is LB._nodes['IntLit']
    assert LA._nodes['Add'] is LB._nodes['Add']

    ok("language_registration")


# ---------------------------------------------------------------------------
# Test 2: Auto-instantiation as dataclasses
# ---------------------------------------------------------------------------

def test_dataclass_construction():
    lit = LA._nodes['IntLit'](value=42)
    assert lit.value == 42
    assert dataclasses.is_dataclass(lit)

    add = LA._nodes['Add'](left=lit, right=lit)
    assert add.left is lit

    ok("dataclass_construction")


# ---------------------------------------------------------------------------
# Test 3: Language.validate
# ---------------------------------------------------------------------------

def test_validate():
    IntLit = LA._nodes['IntLit']
    Add = LA._nodes['Add']
    BoolLit = LA._nodes['BoolLit']

    # Valid LA node
    tree = Add(left=IntLit(value=1), right=IntLit(value=2))
    LA.validate(tree)  # should not raise

    # BoolLit is valid in LA but not in LB
    bl = BoolLit(value=True)
    LA.validate(bl)  # fine
    try:
        LB.validate(bl)
        fail("validate_LB_BoolLit", "should have raised ValidationError")
    except ValidationError:
        pass

    ok("validate")


# ---------------------------------------------------------------------------
# Test 4: Pass expander — identity visitors generated for shared nodes
# ---------------------------------------------------------------------------

def test_pass_expander_shared_nodes():
    class RemoveBools(Pass, source=LA, target=LB):
        # BoolLit not in LB — must handle it explicitly
        def visit_BoolLit(self, node):
            # Replace True -> Not(Not(IntLit(0))), False -> IntLit(0)
            zero = LB._nodes['IntLit'](value=0)
            if node.value:
                return LB._nodes['Not'](operand=LB._nodes['Not'](operand=zero))
            return zero

        # Expr and ListOf require explicit visitors too since they're in LA
        # and their exact classes ARE in LB._nodes (inherited), so expander
        # should auto-generate them.

    # Check that expander generated identity visitors for shared nodes
    assert hasattr(RemoveBools, 'visit_IntLit'), "expander should generate visit_IntLit"
    assert hasattr(RemoveBools, 'visit_Add'), "expander should generate visit_Add"
    assert hasattr(RemoveBools, 'visit_If'), "expander should generate visit_If"
    assert hasattr(RemoveBools, 'visit_Seq'), "expander should generate visit_Seq"
    assert hasattr(RemoveBools, 'visit_ListOf'), "expander should generate visit_ListOf"

    ok("pass_expander_shared_nodes")


# ---------------------------------------------------------------------------
# Test 5: Pass execution — identity traversal works correctly
# ---------------------------------------------------------------------------

def test_pass_identity_traversal():
    IntLit = LA._nodes['IntLit']
    Add = LA._nodes['Add']

    class IdentityPass(Pass, source=LA, target=LA):
        pass  # all visitors auto-generated

    tree = Add(left=IntLit(value=3), right=IntLit(value=4))
    result = IdentityPass()(tree)
    assert result is not tree, "should produce a new node"
    assert result.left.value == 3
    assert result.right.value == 4

    ok("pass_identity_traversal")


# ---------------------------------------------------------------------------
# Test 6: Pass execution — transformation applied
# ---------------------------------------------------------------------------

def test_pass_transformation():
    IntLit = LA._nodes['IntLit']
    BoolLit = LA._nodes['BoolLit']
    Add = LA._nodes['Add']
    Not = LB._nodes['Not']

    class RemoveBools(Pass, source=LA, target=LB):
        def visit_BoolLit(self, node):
            zero = LB._nodes['IntLit'](value=0)
            if node.value:
                return Not(operand=Not(operand=zero))
            return zero

        # Expr node is abstract (no fields), should be fine
        def visit_Expr(self, node):
            return node

    # Add(BoolLit(True), IntLit(5)) -> Add(Not(Not(IntLit(0))), IntLit(5))
    tree = Add(
        left=BoolLit(value=True),
        right=IntLit(value=5),
    )
    result = RemoveBools()(tree)

    assert type(result).__name__ == 'Add'
    assert type(result.left).__name__ == 'Not'
    assert type(result.left.operand).__name__ == 'Not'
    assert type(result.left.operand.operand).__name__ == 'IntLit'
    assert result.left.operand.operand.value == 0
    assert result.right.value == 5

    ok("pass_transformation")


# ---------------------------------------------------------------------------
# Test 7: List field recursion
# ---------------------------------------------------------------------------

def test_list_field_recursion():
    IntLit = LA._nodes['IntLit']
    BoolLit = LA._nodes['BoolLit']
    ListOf = LA._nodes['ListOf']

    class BumpInts(Pass, source=LA, target=LA):
        def visit_IntLit(self, node):
            return IntLit(value=node.value + 1)
        def visit_BoolLit(self, node):
            return node
        def visit_Expr(self, node):
            return node

    tree = ListOf(items=[IntLit(value=1), IntLit(value=2), BoolLit(value=False)])
    result = BumpInts()(tree)

    assert result.items[0].value == 2
    assert result.items[1].value == 3
    assert result.items[2].value == False  # BoolLit unchanged

    ok("list_field_recursion")


# ---------------------------------------------------------------------------
# Test 8: DispatchError when visitor is missing
# ---------------------------------------------------------------------------

def test_dispatch_error():
    IntLit = LA._nodes['IntLit']
    Add = LA._nodes['Add']
    BoolLit = LA._nodes['BoolLit']

    class NoBooleansHandled(Pass, source=LA, target=LB):
        # Intentionally omit visit_BoolLit
        def visit_Expr(self, node):
            return node

    tree = BoolLit(value=True)
    try:
        NoBooleansHandled()(tree)
        fail("dispatch_error", "should have raised DispatchError")
    except DispatchError as e:
        assert 'BoolLit' in str(e)

    ok("dispatch_error")


# ---------------------------------------------------------------------------
# Test 9: Language.unparse
# ---------------------------------------------------------------------------

def test_unparse():
    IntLit = LA._nodes['IntLit']
    Add = LA._nodes['Add']

    tree = Add(left=IntLit(value=1), right=IntLit(value=2))
    s = LA.unparse(tree)
    assert 'Add' in s
    assert 'IntLit' in s
    assert '1' in s
    assert '2' in s

    ok("unparse")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running nanopass framework tests...")
    test_language_registration()
    test_dataclass_construction()
    test_validate()
    test_pass_expander_shared_nodes()
    test_pass_identity_traversal()
    test_pass_transformation()
    test_list_field_recursion()
    test_dispatch_error()
    test_unparse()
    print("All tests passed.")
