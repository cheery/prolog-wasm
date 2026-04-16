"""Intermediate language tower for the Prolog-to-WASM nanopass compiler.

Defines four intermediate languages:

  L1  Normalized Prolog  (core term language, no syntactic sugar)
  L0  Surface Prolog AST (L1 + List/BinOp/UnaryOp, exactly what prolog_parser produces)
  L2  Typed WAM instructions  (one dataclass per instruction)
  L3  Resolved WAM  (L2 with atoms/functors/labels replaced by integer IDs)

The pass tower is:
  parse        text  -> L0
  NormPasses   L0    -> L1   (desugar lists / binops / unary)
  CompileWAM   L1    -> L2   (Prolog term -> WAM instructions)
  InternSyms   L2    -> L3   (str constants / labels -> int IDs)
  EmitWASM     L3    -> CompiledModule  (WAM -> WASM function bodies)
"""

from nanopass import Language

# Pull in the existing parser dataclasses so L0/L1 can reference them
import prolog_parser as _pp


# ===========================================================================
# L1 — Normalized Prolog (base language)
# ===========================================================================

class L1(Language):
    """Core Prolog term language after desugaring.

    Only Atom | Number | Var | Compound remain as term types.
    L1.validate() rejects any List/BinOp/UnaryOp nodes.
    """
    Atom     = _pp.Atom
    Number   = _pp.Number
    Var      = _pp.Var
    Compound = _pp.Compound
    Fact     = _pp.Fact
    Rule     = _pp.Rule
    Query    = _pp.Query
    Program  = _pp.Program


# ===========================================================================
# L0 — Surface Prolog AST  (L1 + syntactic sugar)
# ===========================================================================

class L0(Language, extends=L1):
    """Parser output, verbatim.  Adds the three sugar forms to L1.

    The normalize pass transforms these into L1:
      - List(heads, tail)         -> Compound('.', [head, tail]) / Atom('[]')
      - BinOp(op, left, right)    -> Compound(op, [left, right])
      - UnaryOp(op, operand)      -> Compound(op, [operand])
    """
    List    = _pp.List
    BinOp   = _pp.BinOp
    UnaryOp = _pp.UnaryOp


# ===========================================================================
# L2 — Typed WAM instructions
# ===========================================================================
#
# Replaces the single untyped WAMInstruction(opcode, args) with one
# dataclass per instruction, giving named fields and static structure.
#
# Container nodes:
#   Program2   — full compiled program (predicates + queries)
#   Predicate2 — all clauses for one predicate
#   Clause2    — one clause: a label + instruction sequence
#   Query2     — one query: instructions + variable register map
#
# Instruction nodes are grouped below by category.

class L2(Language):
    """Typed WAM instruction language."""

    # ------------------------------------------------------------------
    # Container nodes
    # ------------------------------------------------------------------

    class Program2:
        predicates: list   # list[Predicate2]
        queries:    list   # list[Query2]

    class Predicate2:
        name:    str
        arity:   int
        clauses: list      # list[Clause2]

    class Clause2:
        label:  str
        instrs: list       # list[<Instr>]

    class Query2:
        instrs:  list      # list[<Instr>]
        reg_map: dict      # {var_name: register_str}

    # ------------------------------------------------------------------
    # HEAD MATCHING  (get_*)
    # ------------------------------------------------------------------

    class GetVariable:
        """get_variable Vn, Ai — first occurrence of var in head arg pos."""
        reg: str   # "X3" or "Y1"
        ai:  int

    class GetValue:
        """get_value Vn, Ai — subsequent occurrence of var in head."""
        reg: str
        ai:  int

    class GetStructure:
        """get_structure f/n, Ai — match/build compound term in head."""
        functor: str
        arity:   int
        ai:      int

    class GetList:
        """get_list Ai — match/build list cell in head."""
        ai: int

    class GetConstant:
        """get_constant c, Ai — match atom or number in head."""
        value: object   # str (atom name) or int/float
        ai:    int

    # ------------------------------------------------------------------
    # SUBTERM UNIFICATION  (unify_*)
    # ------------------------------------------------------------------

    class UnifyVariable:
        """unify_variable Vn — first occurrence inside structure."""
        reg: str

    class UnifyValue:
        """unify_value Vn — subsequent occurrence inside structure."""
        reg: str

    class UnifyLocalValue:
        """unify_local_value Vn — subsequent, may need globalizing."""
        reg: str

    class UnifyConstant:
        """unify_constant c — atom or number inside structure."""
        value: object   # str or int/float

    class UnifyVoid:
        """unify_void n — n anonymous variables."""
        n: int

    # ------------------------------------------------------------------
    # BODY BUILDING  (put_*)
    # ------------------------------------------------------------------

    class PutVariable:
        """put_variable Vn, Ai — first occurrence of var in body goal."""
        reg: str
        ai:  int

    class PutValue:
        """put_value Vn, Ai — subsequent occurrence."""
        reg: str
        ai:  int

    class PutUnsafeValue:
        """put_unsafe_value Yn, Ai — last use of permanent var."""
        reg: str
        ai:  int

    class PutStructure:
        """put_structure f/n, Ai — build compound term."""
        functor: str
        arity:   int
        ai:      int

    class PutList:
        """put_list Ai — build list cell."""
        ai: int

    class PutConstant:
        """put_constant c, Ai — atom or number."""
        value: object   # str or int/float
        ai:    int

    # ------------------------------------------------------------------
    # STRUCTURE BUILDING  (set_*)
    # ------------------------------------------------------------------

    class SetVariable:
        """set_variable Vn — first occurrence inside structure being built."""
        reg: str

    class SetValue:
        """set_value Vn — subsequent occurrence."""
        reg: str

    class SetLocalValue:
        """set_local_value Vn — subsequent, may need globalizing."""
        reg: str

    class SetConstant:
        """set_constant c — atom or number."""
        value: object   # str or int/float

    class SetVoid:
        """set_void n — n anonymous variables."""
        n: int

    # ------------------------------------------------------------------
    # CONTROL
    # ------------------------------------------------------------------

    class Allocate:
        """allocate n — create environment with n permanent vars."""
        n: int

    class Deallocate:
        """deallocate — discard current environment."""

    class Call:
        """call f/n — non-tail call, saves continuation."""
        functor: str
        arity:   int

    class Execute:
        """execute f/n — tail call, no continuation saved."""
        functor: str
        arity:   int

    class Proceed:
        """proceed — return from clause."""

    # ------------------------------------------------------------------
    # CHOICE POINTS
    # ------------------------------------------------------------------

    class TryMeElse:
        """try_me_else L — first clause, save choice point."""
        next_label: str
        arity:      int

    class RetryMeElse:
        """retry_me_else L — middle clause, update choice point."""
        next_label: str
        arity:      int

    class TrustMe:
        """trust_me — last clause, discard choice point."""
        arity: int

    # ------------------------------------------------------------------
    # CUT
    # ------------------------------------------------------------------

    class NeckCut:
        """neck_cut — shallow cut after head."""

    class GetLevel:
        """get_level Yn — save current B into permanent var."""
        reg: str

    class Cut:
        """cut Yn — deep cut using saved B level."""
        reg: str


# ===========================================================================
# L3 — Resolved WAM
# ===========================================================================
#
# Extends L2 by replacing human-readable string references with integers:
#
#   - Atom/functor constants encoded via SymbolTable.encode_constant / functor_pack
#   - Functor fields (str + arity) merged into a single packed int
#   - call/execute targets become WASM function indices
#   - try_me_else / retry_me_else next-clause labels become WASM function indices
#
# Nodes that are UNCHANGED from L2 inherit the same Python class, so the
# pass expander auto-generates identity visitors for them in InternSymbols.
# The 10 nodes redefined here require explicit visit_* methods in the pass.

class L3(Language, extends=L2):
    """WAM with all symbolic references resolved to integers."""

    # GetConstant: value is now an encoded i32, not a raw Python value
    class GetConstant:
        """get_constant c, Ai — c is a SymbolTable-encoded i32."""
        value: int
        ai:    int

    # GetStructure: functor+arity merged into a packed i32
    class GetStructure:
        """get_structure f/n, Ai — functor_packed = syms.functor_pack(name, arity)."""
        functor_packed: int
        ai:             int

    # PutConstant / PutStructure: same pattern as the Get versions
    class PutConstant:
        """put_constant c, Ai — c is a SymbolTable-encoded i32."""
        value: int
        ai:    int

    class PutStructure:
        """put_structure f/n, Ai — functor_packed = syms.functor_pack(name, arity)."""
        functor_packed: int
        ai:             int

    # UnifyConstant / SetConstant
    class UnifyConstant:
        """unify_constant c — c is a SymbolTable-encoded i32."""
        value: int

    class SetConstant:
        """set_constant c — c is a SymbolTable-encoded i32."""
        value: int

    # Call / Execute: (functor, arity) -> WASM function index
    class Call:
        """call target — func_index is a WASM function index."""
        func_index: int

    class Execute:
        """execute target — func_index is a WASM function index."""
        func_index: int

    # TryMeElse / RetryMeElse: label -> WASM function index
    class TryMeElse:
        """try_me_else L — next_func_index is the WASM idx of the retry clause."""
        next_func_index: int
        arity:           int

    class RetryMeElse:
        """retry_me_else L — next_func_index is the WASM idx of the retry clause."""
        next_func_index: int
        arity:           int
