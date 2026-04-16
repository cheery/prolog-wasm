"""Nanopass framework for Python.

Provides Language and Pass base classes for implementing nanopass compilers.
Each Language defines a set of intermediate-representation node types.
Passes transform programs from one language to another; the pass expander
automatically generates identity traversal for unchanged node types.

Usage
-----
Define a language with inner class definitions (auto-converted to dataclasses):

    class L0(Language):
        class Program:
            clauses: list

    class L1(Language, extends=L0):
        remove = ['List']       # node types removed from L0
        class NewNode:          # additional node type added to L1
            value: int

You can also register existing dataclass types by assigning them:

    from some_module import MyNode
    class L0(Language):
        MyNode = MyNode         # registered as-is

Define a pass:

    class NormalizeLists(Pass, source=L0, target=L1):
        def visit_List(self, node):
            ...                 # explicit transformer for List -> something

    result = NormalizeLists()(root_node)

The pass expander auto-generates identity visitors for nodes whose exact Python
class appears under the same name in both source and target.  For nodes only in
the source language, a DispatchError is raised at runtime if visit() is called.
"""

import dataclasses

__all__ = [
    'Language', 'Pass',
    'LanguageError', 'ValidationError', 'PassDefinitionError', 'DispatchError',
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LanguageError(Exception):
    pass


class ValidationError(LanguageError):
    pass


class PassDefinitionError(Exception):
    pass


class DispatchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_identity_visitor(node_cls):
    """Return a visit_X method that recursively copies a node of node_cls.

    - list fields: map self.visit over dataclass items, pass primitives through
    - dataclass fields: recurse with self.visit
    - everything else: copy as-is
    """
    flds = dataclasses.fields(node_cls)

    def visit_X(self, node):
        new_vals = {}
        for f in flds:
            v = getattr(node, f.name)
            if isinstance(v, list):
                new_vals[f.name] = [
                    self.visit(item) if dataclasses.is_dataclass(item) else item
                    for item in v
                ]
            elif dataclasses.is_dataclass(v):
                new_vals[f.name] = self.visit(v)
            else:
                new_vals[f.name] = v
        return node_cls(**new_vals)

    return visit_X


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

class Language:
    """Base class for intermediate language definitions.

    Subclass to define a language.  Inner class definitions are
    auto-converted to dataclasses and registered as node types.
    Existing dataclasses assigned as class attributes are registered as-is.

    Language inheritance:

        class L1(Language, extends=L0):
            remove = ['NodeToRemove']   # list of node-type names to drop
            class NewNode:              # new node type to add
                field: str
    """

    _nodes: dict  # name -> dataclass type; populated by __init_subclass__

    def __init_subclass__(cls, extends=None, **kwargs):
        super().__init_subclass__(**kwargs)

        # Seed from parent language if extending
        if extends is not None:
            cls._nodes = dict(extends._nodes)
        else:
            cls._nodes = {}

        # Apply declared removals
        for name in vars(cls).get('remove', []):
            cls._nodes.pop(name, None)

        # Register inner classes / assigned dataclasses
        for name, val in vars(cls).items():
            if name.startswith('_') or name == 'remove':
                continue
            if not isinstance(val, type):
                continue
            if dataclasses.is_dataclass(val):
                # Pre-existing dataclass — register as-is
                cls._nodes[name] = val
            else:
                # Raw class definition — convert to dataclass and register
                dc = dataclasses.dataclass(val)
                cls._nodes[name] = dc
                setattr(cls, name, dc)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate(cls, node, _path=''):
        """Recursively validate that *node* is well-formed for this language.

        Raises ValidationError with a path description if any node type is
        not registered in this language.
        """
        if not dataclasses.is_dataclass(node):
            raise ValidationError(
                f"{_path or 'root'}: expected dataclass node, "
                f"got {type(node).__name__!r}"
            )
        name = type(node).__name__
        if name not in cls._nodes:
            raise ValidationError(
                f"{_path or 'root'}: {name!r} is not a node type "
                f"in language {cls.__name__}"
            )
        for f in dataclasses.fields(node):
            v = getattr(node, f.name)
            path = f"{_path}.{f.name}" if _path else f.name
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if dataclasses.is_dataclass(item):
                        cls.validate(item, f"{path}[{i}]")
            elif dataclasses.is_dataclass(v):
                cls.validate(v, path)

    # ------------------------------------------------------------------
    # Unparser (debugging)
    # ------------------------------------------------------------------

    @classmethod
    def unparse(cls, node, _indent=0) -> str:
        """Return a readable string representation of *node*."""
        if not dataclasses.is_dataclass(node):
            return repr(node)
        name = type(node).__name__
        flds = dataclasses.fields(node)
        if not flds:
            return f"{name}()"
        parts = []
        for f in flds:
            v = getattr(node, f.name)
            if isinstance(v, list):
                items = [
                    cls.unparse(item, _indent + 4)
                    if dataclasses.is_dataclass(item)
                    else repr(item)
                    for item in v
                ]
                parts.append(f"{f.name}=[{', '.join(items)}]")
            elif dataclasses.is_dataclass(v):
                parts.append(f"{f.name}={cls.unparse(v, _indent + 2)}")
            else:
                parts.append(f"{f.name}={v!r}")
        return f"{name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

class Pass:
    """Base class for compiler passes.

    Subclass with source= and target= to define a pass:

        class MyPass(Pass, source=L0, target=L1):
            def visit_List(self, node):
                ...

    The pass expander auto-generates identity visitors for every node type
    whose *exact Python class* appears under the same name in both source and
    target.  If that identity visitor would recurse into child nodes, it calls
    self.visit() on each dataclass child.

    For source nodes that are absent from the target (or mapped to a different
    Python class in the target), the expander generates nothing.  Calling
    self.visit() on such a node raises DispatchError unless you define an
    explicit visit_X method.

    Invoke a pass by calling it:

        output = MyPass()(root)
    """

    source: 'Language'
    target: 'Language'

    def __init_subclass__(cls, source=None, target=None, **kwargs):
        super().__init_subclass__(**kwargs)

        if source is None or target is None:
            return  # intermediate abstract base — skip expander

        cls.source = source
        cls.target = target

        src_nodes = source._nodes
        tgt_nodes = target._nodes

        for name, src_cls in src_nodes.items():
            method_name = f'visit_{name}'
            if method_name in vars(cls):
                continue  # explicit override — leave it alone

            # Auto-generate identity visitor only when the exact same Python
            # class is registered under the same name in the target language.
            if tgt_nodes.get(name) is src_cls:
                visitor = _make_identity_visitor(src_cls)
                visitor.__name__ = method_name
                visitor.__qualname__ = f'{cls.__name__}.{method_name}'
                setattr(cls, method_name, visitor)
            # Otherwise: no visitor generated; DispatchError at runtime.

    def visit(self, node):
        """Dispatch to the appropriate visit_X method for *node*."""
        if not dataclasses.is_dataclass(node):
            # Primitive value — pass through
            return node
        cls_name = type(node).__name__
        method = getattr(self, f'visit_{cls_name}', None)
        if method is None:
            raise DispatchError(
                f"{type(self).__name__} has no visitor for node type "
                f"{cls_name!r}. Add a visit_{cls_name}() method or ensure "
                f"{cls_name} is shared between source and target languages."
            )
        return method(node)

    def __call__(self, root):
        return self.visit(root)
