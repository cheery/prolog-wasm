"""Prolog-to-WAM compiler.

Takes AST nodes from prolog_parser and compiles them into WAM instructions.
The output is a list of labeled instruction sequences that can be:
  1. Executed directly by the WAM interpreter (wam.py)
  2. Compiled further to WASM (next step)

WAM instruction set used (following Ait-Kaci's tutorial):

  Head matching (get_*):
    get_variable    Vn, Ai    -- first occurrence of var in head arg position
    get_value       Vn, Ai    -- subsequent occurrence
    get_structure   f/n, Ai   -- match/build compound term
    get_list        Ai        -- match/build list (./2)
    get_constant    c, Ai     -- match atom or number

  Subterm (unify_*):
    unify_variable  Vn        -- first occurrence inside structure
    unify_value     Vn        -- subsequent occurrence
    unify_local_value Vn      -- subsequent, might need globalizing
    unify_constant  c         -- atom or number inside structure
    unify_void      n         -- anonymous variable(s)

  Body building (put_*):
    put_variable    Vn, Ai    -- first occurrence of var in body goal
    put_value       Vn, Ai    -- subsequent occurrence
    put_unsafe_value Yn, Ai   -- last use of permanent var (needs globalizing)
    put_structure   f/n, Ai   -- build compound term
    put_list        Ai        -- build list cell
    put_constant    c, Ai     -- atom or number

  Structure building (set_*):
    set_variable    Vn        -- first occurrence inside structure being built
    set_value       Vn        -- subsequent occurrence
    set_local_value Vn        -- subsequent, might need globalizing
    set_constant    c         -- atom or number
    set_void        n         -- anonymous variable(s)

  Control:
    allocate        n         -- create environment with n permanent vars
    deallocate                -- discard current environment
    call            f/n       -- call predicate
    execute         f/n       -- tail-call predicate
    proceed                   -- return from clause

  Choice:
    try_me_else     L         -- first clause, save choice point
    retry_me_else   L         -- middle clause, update choice point
    trust_me                  -- last clause, discard choice point

  Cut:
    neck_cut                  -- cut after head (shallow)
    get_level       Yn        -- save B0 into permanent var
    cut             Yn        -- deep cut using saved B0
"""

from dataclasses import dataclass, field
from typing import Any
from prolog_parser import (
    Atom, Number, Var, Compound, List, BinOp, UnaryOp,
    Fact, Rule, Query, Program, parse,
)
from languages import L2


# ---------------------------------------------------------------------------
# WAM instruction representation
# ---------------------------------------------------------------------------

@dataclass
class WAMInstruction:
    opcode: str
    args: list = field(default_factory=list)

    def __repr__(self):
        if self.args:
            parts = []
            for a in self.args:
                if isinstance(a, tuple):
                    name, arity = a
                    parts.append(f"{name}/{arity}")
                else:
                    parts.append(repr(a))
            return f"{self.opcode} {', '.join(parts)}"
        return self.opcode


@dataclass
class CompiledPredicate:
    """All compiled clauses for one predicate."""
    name: str
    arity: int
    clauses: list  # list of (label, [WAMInstruction])


# ---------------------------------------------------------------------------
# Flattening: convert nested terms to register-level representation
# ---------------------------------------------------------------------------

def flatten_term(term):
    """Walk a term depth-first, extracting sub-terms that need registers.

    Returns a list of (register, directive) pairs where directive is one of:
      ('structure', functor, arity, [arg_registers])
      ('list', head_reg, tail_reg)
      ('constant', value)
      ('variable', name)
      ('integer', value)
    """
    # This is used internally by the register allocator.
    pass


# ---------------------------------------------------------------------------
# Variable analysis
# ---------------------------------------------------------------------------

def collect_vars(term, vs: set):
    """Collect all variable names in a term."""
    if isinstance(term, Var):
        vs.add(term.name)
    elif isinstance(term, Compound):
        for a in term.args:
            collect_vars(a, vs)
    elif isinstance(term, List):
        for h in term.heads:
            collect_vars(h, vs)
        if term.tail is not None:
            collect_vars(term.tail, vs)
    elif isinstance(term, BinOp):
        collect_vars(term.left, vs)
        collect_vars(term.right, vs)
    elif isinstance(term, UnaryOp):
        collect_vars(term.operand, vs)


def vars_in(term) -> set:
    vs = set()
    collect_vars(term, vs)
    return vs


def flatten_body(body) -> list:
    """Flatten a body (list of goals) — it's already a list from the parser."""
    return body


# ---------------------------------------------------------------------------
# Register allocation
# ---------------------------------------------------------------------------

def allocate_registers(head, body_goals):
    """Decide which variables get temporary (Xi) vs permanent (Yi) registers.

    Permanent variables: those that appear in more than one body goal,
    or that appear in both a non-first body goal and the head.
    These must survive across calls.

    Returns (reg_map: {var_name -> "Xi" or "Yi"}, num_permanent: int)
    """
    # Collect head variables
    head_vars = vars_in(head) if head else set()

    # Collect variables per body goal
    goal_var_sets = []
    for g in body_goals:
        goal_var_sets.append(vars_in(g))

    # All variables
    all_vars = set(head_vars)
    for gvs in goal_var_sets:
        all_vars.update(gvs)
    all_vars.discard("_")  # anonymous vars don't get registers

    if not body_goals:
        # Fact: everything is temporary
        reg_map = {}
        head_arity = len(head.args) if isinstance(head, Compound) else 0
        counter = head_arity
        for arg in (head.args if isinstance(head, Compound) else []):
            if isinstance(arg, Var) and arg.name != "_" and arg.name not in reg_map:
                # Bare variable in head arg position gets argument register
                reg_map[arg.name] = f"X{head.args.index(arg) + 1}"
        # Sub-variables inside structures
        for i, arg in enumerate(head.args if isinstance(head, Compound) else []):
            if not isinstance(arg, Var):
                for v in sorted(vars_in(arg)):
                    if v != "_" and v not in reg_map:
                        counter += 1
                        reg_map[v] = f"X{counter}"
        return reg_map, 0

    # Determine permanent variables:
    # A variable is permanent if it appears in two or more body goals,
    # or if it appears in a body goal that is not the first AND in the head.
    # Simplified: permanent = appears after the first goal boundary (goal index > 0)
    # AND also appears somewhere before or in a different goal.

    # More precisely: permanent vars are those that need to survive across a call.
    # A var is permanent if it occurs in goal[i] and goal[j] where i != j,
    # or in the head and in goal[i>0].
    permanent = set()

    # Count in how many "chunks" each variable appears.
    # Chunks: head is chunk 0, goal[0] is chunk 1, goal[1] is chunk 2, etc.
    # A variable is permanent if it appears in 2+ chunks AND at least one is > 0.
    chunk_presence = {}
    for v in all_vars:
        chunks = set()
        if v in head_vars:
            chunks.add(0)
        for gi, gvs in enumerate(goal_var_sets):
            if v in gvs:
                chunks.add(gi + 1)
        chunk_presence[v] = chunks

    for v, chunks in chunk_presence.items():
        if len(chunks) >= 2:
            # If variable appears only in head (chunk 0) and first goal (chunk 1),
            # it can still be temporary IF it's a bare head argument.
            # But to be safe for now: mark as permanent if spans 2+ chunks
            # and at least one chunk is > 0 (i.e., it's in at least one body goal)
            if max(chunks) > 0:
                # Check: if it only spans head + first_goal, it can be temp
                # if it's a bare head arg (since it's in Ai already).
                if chunks == {0, 1}:
                    # Only head and first goal — can be temp if bare head arg
                    head_arity = len(head.args) if isinstance(head, Compound) else 0
                    is_bare_head = False
                    if isinstance(head, Compound):
                        for i, arg in enumerate(head.args):
                            if isinstance(arg, Var) and arg.name == v:
                                is_bare_head = True
                    if is_bare_head:
                        continue  # stays temporary
                permanent.add(v)

    # Assign permanent registers (Y1, Y2, ...)
    perm_list = sorted(permanent)
    num_perm = len(perm_list)

    reg_map = {}
    for i, v in enumerate(perm_list):
        reg_map[v] = f"Y{i + 1}"

    # Assign temporary registers
    head_arity = len(head.args) if isinstance(head, Compound) else 0
    counter = head_arity

    # Head args: bare variables get argument registers
    if isinstance(head, Compound):
        for i, arg in enumerate(head.args):
            if isinstance(arg, Var) and arg.name != "_" and arg.name not in reg_map:
                reg_map[arg.name] = f"X{i + 1}"

    # Sub-variables inside head structures
    if isinstance(head, Compound):
        for arg in head.args:
            if not isinstance(arg, Var):
                for v in sorted(vars_in(arg)):
                    if v != "_" and v not in reg_map:
                        counter += 1
                        reg_map[v] = f"X{counter}"

    # Body-only variables
    for v in sorted(all_vars):
        if v != "_" and v not in reg_map:
            counter += 1
            reg_map[v] = f"X{counter}"

    return reg_map, num_perm


# ---------------------------------------------------------------------------
# Term normalization: convert parser AST to a uniform representation
# ---------------------------------------------------------------------------

def normalize_term(term):
    """Convert List nodes to Compound('.', [head, tail]) form for WAM compilation."""
    if isinstance(term, List):
        # Build ./2 chain
        if term.tail is not None:
            tail = normalize_term(term.tail)
        else:
            tail = Atom("[]")
        for h in reversed(term.heads):
            tail = Compound(".", [normalize_term(h), tail])
        return tail
    elif isinstance(term, Compound):
        return Compound(term.functor, [normalize_term(a) for a in term.args])
    elif isinstance(term, BinOp):
        return Compound(term.op, [normalize_term(term.left), normalize_term(term.right)])
    elif isinstance(term, UnaryOp):
        if term.op == "-":
            return Compound("-", [normalize_term(term.operand)])
        return Compound(term.op, [normalize_term(term.operand)])
    return term


# ---------------------------------------------------------------------------
# Instruction emission
# ---------------------------------------------------------------------------

class ClauseCompiler:
    """Compiles a single clause (fact or rule) to WAM instructions."""

    def __init__(self, head, body_goals, reg_map, num_perm):
        self.head = head
        self.body_goals = body_goals
        self.reg_map = reg_map
        self.num_perm = num_perm
        self.instrs = []
        self.seen = set()  # tracks first-occurrence variables across the clause

    def compile(self) -> list[WAMInstruction]:
        # Allocate environment if we have permanent variables
        if self.body_goals and self.num_perm > 0:
            self.instrs.append(WAMInstruction("allocate", [self.num_perm]))

        # Head matching
        if isinstance(self.head, Compound):
            for i, arg in enumerate(self.head.args):
                self._emit_get(arg, i + 1)
        elif isinstance(self.head, Atom):
            pass  # atom/0 head: nothing to match

        # Body goals
        if self.body_goals:
            for gi, goal in enumerate(self.body_goals):
                is_last = (gi == len(self.body_goals) - 1)
                self._emit_goal(goal, is_last)
        else:
            # Fact: just proceed
            self.instrs.append(WAMInstruction("proceed"))

        return self.instrs

    # -- Head matching --

    def _emit_get(self, arg, ai):
        """Emit get_* instruction for head argument at position ai."""
        if isinstance(arg, Var):
            if arg.name == "_":
                return  # anonymous: don't match
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name in self.seen:
                self.instrs.append(WAMInstruction("get_value", [vn, ai]))
            else:
                self.instrs.append(WAMInstruction("get_variable", [vn, ai]))
                self.seen.add(arg.name)

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(WAMInstruction("get_constant", [val, ai]))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(WAMInstruction("get_list", [ai]))
            self._emit_unify(arg.args[0])
            self._emit_unify(arg.args[1])

        elif isinstance(arg, Compound):
            functor = (arg.functor, len(arg.args))
            self.instrs.append(WAMInstruction("get_structure", [functor, ai]))
            for sub in arg.args:
                self._emit_unify(sub)

    # -- Subterm unification (inside structures/lists in head) --

    def _emit_unify(self, sub):
        """Emit unify_* instruction for a subterm."""
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(WAMInstruction("unify_void", [1]))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(WAMInstruction("unify_void", [1]))
            elif sub.name not in self.seen:
                self.instrs.append(WAMInstruction("unify_variable", [vn]))
                self.seen.add(sub.name)
            else:
                self.instrs.append(WAMInstruction("unify_value", [vn]))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(WAMInstruction("unify_constant", [val]))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            # Nested list inside a structure: allocate a temp register
            # and emit get_list later. For now, use unify_variable + get_list.
            temp = self._alloc_temp()
            self.instrs.append(WAMInstruction("unify_variable", [temp]))
            # We'll need to emit get_list for this temp after the current structure.
            # For simplicity, defer it.
            self._deferred_gets.append((sub, temp))

        elif isinstance(sub, Compound):
            # Nested compound: allocate temp, unify_variable, then get_structure
            temp = self._alloc_temp()
            self.instrs.append(WAMInstruction("unify_variable", [temp]))
            self._deferred_gets.append((sub, temp))

    _deferred_gets = []
    _temp_counter = 100  # high enough to avoid collision

    def _alloc_temp(self):
        ClauseCompiler._temp_counter += 1
        return f"X{ClauseCompiler._temp_counter}"

    # -- Body goals --

    def _emit_goal(self, goal, is_last):
        """Emit put_* + call/execute for a body goal."""
        goal = normalize_term(goal)

        # Cut
        if isinstance(goal, Atom) and goal.name == "!":
            self.instrs.append(WAMInstruction("neck_cut"))
            return

        if isinstance(goal, Compound):
            functor = goal.functor
            arity = len(goal.args)

            # Emit put_* for each argument
            for i, arg in enumerate(goal.args):
                self._emit_put(arg, i + 1)

            pred = (functor, arity)
        elif isinstance(goal, Atom):
            pred = (goal.name, 0)
        else:
            return  # skip unrecognized goals

        if is_last:
            has_perm = self.num_perm > 0
            if has_perm:
                self.instrs.append(WAMInstruction("deallocate"))
            self.instrs.append(WAMInstruction("execute", [pred]))
        else:
            self.instrs.append(WAMInstruction("call", [pred]))

    # -- Body argument building --

    def _emit_put(self, arg, ai):
        """Emit put_* instruction for body goal argument at position ai."""
        if isinstance(arg, Var):
            if arg.name == "_":
                # Anonymous: create a fresh variable
                self.instrs.append(WAMInstruction("put_variable", [f"X{ai}", ai]))
                return
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name not in self.seen:
                self.seen.add(arg.name)
                self.instrs.append(WAMInstruction("put_variable", [vn, ai]))
            else:
                if vn.startswith("Y"):
                    self.instrs.append(WAMInstruction("put_value", [vn, ai]))
                else:
                    self.instrs.append(WAMInstruction("put_value", [vn, ai]))

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(WAMInstruction("put_constant", [val, ai]))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(WAMInstruction("put_list", [ai]))
            self._emit_set(arg.args[0])
            self._emit_set(arg.args[1])

        elif isinstance(arg, Compound):
            functor = (arg.functor, len(arg.args))
            self.instrs.append(WAMInstruction("put_structure", [functor, ai]))
            for sub in arg.args:
                self._emit_set(sub)

    def _emit_set(self, sub):
        """Emit set_* instruction for building a subterm in body."""
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(WAMInstruction("set_void", [1]))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(WAMInstruction("set_void", [1]))
            elif sub.name not in self.seen:
                self.seen.add(sub.name)
                self.instrs.append(WAMInstruction("set_variable", [vn]))
            else:
                self.instrs.append(WAMInstruction("set_value", [vn]))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(WAMInstruction("set_constant", [val]))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            # Nested list: need to build inner list
            # The WAM approach: put_list into a temp, then set_value that temp
            # But since we're already inside a set sequence, we need to use
            # set_variable for a temp, then later fill it with put_list.
            # Simplified: emit inline.
            self._emit_set(sub.args[0])
            self._emit_set(sub.args[1])

        elif isinstance(sub, Compound):
            # Nested structure: simplified — set_void placeholder
            # A full compiler would use auxiliary temp registers
            self.instrs.append(WAMInstruction("set_void", [1]))

        else:
            self.instrs.append(WAMInstruction("set_void", [1]))


# ---------------------------------------------------------------------------
# Query compiler
# ---------------------------------------------------------------------------

class QueryCompiler:
    """Compiles a query (?- goal1, goal2, ...) to WAM instructions."""

    def __init__(self, goals, reg_map):
        self.goals = goals
        self.reg_map = reg_map
        self.instrs = []
        self.seen = set()

    def compile(self) -> list[WAMInstruction]:
        for gi, goal in enumerate(self.goals):
            goal = normalize_term(goal)
            is_last = (gi == len(self.goals) - 1)

            if isinstance(goal, Compound):
                for i, arg in enumerate(goal.args):
                    self._emit_put(arg, i + 1)
                pred = (goal.functor, len(goal.args))
            elif isinstance(goal, Atom):
                pred = (goal.name, 0)
            else:
                continue

            if is_last:
                self.instrs.append(WAMInstruction("execute", [pred]))
            else:
                self.instrs.append(WAMInstruction("call", [pred]))

        return self.instrs

    def _emit_put(self, arg, ai):
        if isinstance(arg, Var):
            if arg.name == "_":
                self.instrs.append(WAMInstruction("put_variable", [f"X{ai}", ai]))
                return
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name not in self.seen:
                self.seen.add(arg.name)
                self.instrs.append(WAMInstruction("put_variable", [vn, ai]))
            else:
                self.instrs.append(WAMInstruction("put_value", [vn, ai]))

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(WAMInstruction("put_constant", [val, ai]))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(WAMInstruction("put_list", [ai]))
            self._emit_set(arg.args[0])
            self._emit_set(arg.args[1])

        elif isinstance(arg, Compound):
            functor = (arg.functor, len(arg.args))
            self.instrs.append(WAMInstruction("put_structure", [functor, ai]))
            for sub in arg.args:
                self._emit_set(sub)

    def _emit_set(self, sub):
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(WAMInstruction("set_void", [1]))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(WAMInstruction("set_void", [1]))
            elif sub.name not in self.seen:
                self.seen.add(sub.name)
                self.instrs.append(WAMInstruction("set_variable", [vn]))
            else:
                self.instrs.append(WAMInstruction("set_value", [vn]))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(WAMInstruction("set_constant", [val]))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            self._emit_set(sub.args[0])
            self._emit_set(sub.args[1])

        elif isinstance(sub, Compound):
            self.instrs.append(WAMInstruction("set_void", [1]))
        else:
            self.instrs.append(WAMInstruction("set_void", [1]))


# ---------------------------------------------------------------------------
# Predicate-level compilation (multi-clause with choice points)
# ---------------------------------------------------------------------------

def compile_predicate(name, arity, clauses) -> CompiledPredicate:
    """Compile all clauses for a predicate, wrapping with choice instructions.

    clauses: list of (head, body_goals) where body_goals is a list (possibly empty).
    """
    compiled_clauses = []

    for ci, (head, body_goals) in enumerate(clauses):
        head = normalize_term(head)
        body_goals = [normalize_term(g) for g in body_goals]

        reg_map, num_perm = allocate_registers(head, body_goals)
        cc = ClauseCompiler(head, body_goals, reg_map, num_perm)
        clause_instrs = cc.compile()

        label = f"{name}/{arity}" if len(clauses) == 1 else f"{name}/{arity}_c{ci}"

        # Prepend choice instruction for multi-clause predicates
        if len(clauses) > 1:
            if ci == 0:
                next_label = f"{name}/{arity}_c{ci + 1}"
                clause_instrs.insert(0, WAMInstruction("try_me_else", [next_label, arity]))
            elif ci < len(clauses) - 1:
                next_label = f"{name}/{arity}_c{ci + 1}"
                clause_instrs.insert(0, WAMInstruction("retry_me_else", [next_label, arity]))
            else:
                clause_instrs.insert(0, WAMInstruction("trust_me", [arity]))

        compiled_clauses.append((label, clause_instrs))

    return CompiledPredicate(name, arity, compiled_clauses)


# ---------------------------------------------------------------------------
# Program compilation
# ---------------------------------------------------------------------------

def compile_program(program: Program):
    """Compile a full Prolog program into WAM instructions.

    Returns:
      predicates: dict of {name/arity: CompiledPredicate}
      queries: list of (query_instrs, reg_map)
    """
    # Group clauses by predicate
    pred_clauses = {}  # (name, arity) -> [(head, body_goals)]
    queries = []

    for clause in program.clauses:
        if isinstance(clause, Fact):
            head = clause.head
            if isinstance(head, Compound):
                key = (head.functor, len(head.args))
            elif isinstance(head, Atom):
                key = (head.name, 0)
            else:
                continue
            pred_clauses.setdefault(key, []).append((head, []))

        elif isinstance(clause, Rule):
            head = clause.head
            if isinstance(head, Compound):
                key = (head.functor, len(head.args))
            elif isinstance(head, Atom):
                key = (head.name, 0)
            else:
                continue
            pred_clauses.setdefault(key, []).append((head, clause.body))

        elif isinstance(clause, Query):
            # Compile query
            all_vars = set()
            for g in clause.goals:
                collect_vars(g, all_vars)
            all_vars.discard("_")

            # Assign registers for query variables
            max_arity = 0
            for g in clause.goals:
                if isinstance(g, Compound):
                    max_arity = max(max_arity, len(g.args))
                elif isinstance(g, BinOp):
                    max_arity = max(max_arity, 2)

            reg_map = {}
            for i, v in enumerate(sorted(all_vars)):
                reg_map[v] = f"X{max_arity + i + 1}"

            qc = QueryCompiler(clause.goals, reg_map)
            query_instrs = qc.compile()
            queries.append((query_instrs, reg_map))

    # Compile each predicate
    predicates = {}
    for (name, arity), clauses in pred_clauses.items():
        pred = compile_predicate(name, arity, clauses)
        predicates[f"{name}/{arity}"] = pred

    return predicates, queries


# ---------------------------------------------------------------------------
# L2 conversion: WAMInstruction -> typed L2 nodes
# ---------------------------------------------------------------------------

def _wam_instr_to_l2(instr):
    """Convert a WAMInstruction to the corresponding L2 typed node."""
    op = instr.opcode
    a = instr.args
    if op == 'get_variable':
        return L2.GetVariable(reg=a[0], ai=a[1])
    elif op == 'get_value':
        return L2.GetValue(reg=a[0], ai=a[1])
    elif op == 'get_structure':
        name, arity = a[0]
        return L2.GetStructure(functor=name, arity=arity, ai=a[1])
    elif op == 'get_list':
        return L2.GetList(ai=a[0])
    elif op == 'get_constant':
        return L2.GetConstant(value=a[0], ai=a[1])
    elif op == 'unify_variable':
        return L2.UnifyVariable(reg=a[0])
    elif op == 'unify_value':
        return L2.UnifyValue(reg=a[0])
    elif op == 'unify_local_value':
        return L2.UnifyLocalValue(reg=a[0])
    elif op == 'unify_constant':
        return L2.UnifyConstant(value=a[0])
    elif op == 'unify_void':
        return L2.UnifyVoid(n=a[0])
    elif op == 'put_variable':
        return L2.PutVariable(reg=a[0], ai=a[1])
    elif op == 'put_value':
        return L2.PutValue(reg=a[0], ai=a[1])
    elif op == 'put_unsafe_value':
        return L2.PutUnsafeValue(reg=a[0], ai=a[1])
    elif op == 'put_structure':
        name, arity = a[0]
        return L2.PutStructure(functor=name, arity=arity, ai=a[1])
    elif op == 'put_list':
        return L2.PutList(ai=a[0])
    elif op == 'put_constant':
        return L2.PutConstant(value=a[0], ai=a[1])
    elif op == 'set_variable':
        return L2.SetVariable(reg=a[0])
    elif op == 'set_value':
        return L2.SetValue(reg=a[0])
    elif op == 'set_local_value':
        return L2.SetLocalValue(reg=a[0])
    elif op == 'set_constant':
        return L2.SetConstant(value=a[0])
    elif op == 'set_void':
        return L2.SetVoid(n=a[0])
    elif op == 'allocate':
        return L2.Allocate(n=a[0])
    elif op == 'deallocate':
        return L2.Deallocate()
    elif op == 'call':
        name, arity = a[0]
        return L2.Call(functor=name, arity=arity)
    elif op == 'execute':
        name, arity = a[0]
        return L2.Execute(functor=name, arity=arity)
    elif op == 'proceed':
        return L2.Proceed()
    elif op == 'try_me_else':
        return L2.TryMeElse(next_label=a[0], arity=a[1])
    elif op == 'retry_me_else':
        return L2.RetryMeElse(next_label=a[0], arity=a[1])
    elif op == 'trust_me':
        return L2.TrustMe(arity=a[0] if a else 0)
    elif op == 'neck_cut':
        return L2.NeckCut()
    elif op == 'get_level':
        return L2.GetLevel(reg=a[0])
    elif op == 'cut':
        return L2.Cut(reg=a[0])
    else:
        raise ValueError(f"Unknown WAM opcode: {op!r}")


def compile_program_l2(program: Program) -> 'L2.Program2':
    """Compile a Prolog program to a Program2 (L2 typed WAM instructions).

    Calls the existing compile_program(), then converts each WAMInstruction
    to the corresponding typed L2 node.  Returns a Program2 containing
    Predicate2 / Clause2 / Query2 nodes.
    """
    predicates, queries = compile_program(program)

    pred_list = []
    for _key, pred in predicates.items():
        clauses_l2 = []
        for label, instrs in pred.clauses:
            l2_instrs = [_wam_instr_to_l2(i) for i in instrs]
            clauses_l2.append(L2.Clause2(label=label, instrs=l2_instrs))
        pred_list.append(
            L2.Predicate2(name=pred.name, arity=pred.arity, clauses=clauses_l2)
        )

    query_list = []
    for instrs, reg_map in queries:
        l2_instrs = [_wam_instr_to_l2(i) for i in instrs]
        query_list.append(L2.Query2(instrs=l2_instrs, reg_map=reg_map))

    return L2.Program2(predicates=pred_list, queries=query_list)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_compiled(predicates, queries):
    """Print compiled WAM instructions in a readable format."""
    for key, pred in sorted(predicates.items()):
        for label, instrs in pred.clauses:
            print(f"{label}:")
            for instr in instrs:
                print(f"    {instr}")
            print()

    for qi, (instrs, reg_map) in enumerate(queries):
        var_info = ", ".join(f"{v}={r}" for v, r in sorted(reg_map.items()))
        print(f"query_{qi} [{var_info}]:")
        for instr in instrs:
            print(f"    {instr}")
        print()


# ---------------------------------------------------------------------------
# Main: test with example programs
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

        % List operations
        member(X, [X | _]).
        member(X, [_ | T]) :- member(X, T).

        append([], L, L).
        append([H | T], L, [H | R]) :- append(T, L, R).

        length([], 0).
        length([_ | T], N) :- length(T, N1), N is N1 + 1.

        ?- grandparent(tom, X).
        ?- member(2, [1, 2, 3]).
    """

    program = parse(test_source)
    predicates, queries = compile_program(program)
    print_compiled(predicates, queries)
