"""Prolog-to-WAM compiler.

Input:  a normalized L1 Program (Atom | Number | Var | Compound terms only,
        no List / BinOp / UnaryOp sugar).
Output: a Program2 (L2 typed WAM instruction nodes).

WAM instruction set: see languages.py for the full L2 node type definitions.
"""

from prolog_parser import Atom, Number, Var, Compound, Fact, Rule, Query, Program
from languages import L2


# ---------------------------------------------------------------------------
# Variable analysis
# ---------------------------------------------------------------------------

def collect_vars(term, vs: set):
    """Collect all variable names in an L1 term."""
    if isinstance(term, Var):
        vs.add(term.name)
    elif isinstance(term, Compound):
        for a in term.args:
            collect_vars(a, vs)


def vars_in(term) -> set:
    vs = set()
    collect_vars(term, vs)
    return vs


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
    head_vars = vars_in(head) if head else set()

    goal_var_sets = [vars_in(g) for g in body_goals]

    all_vars = set(head_vars)
    for gvs in goal_var_sets:
        all_vars.update(gvs)
    all_vars.discard("_")

    if not body_goals:
        # Fact: everything is temporary
        reg_map = {}
        counter = len(head.args) if isinstance(head, Compound) else 0
        for arg in (head.args if isinstance(head, Compound) else []):
            if isinstance(arg, Var) and arg.name != "_" and arg.name not in reg_map:
                reg_map[arg.name] = f"X{head.args.index(arg) + 1}"
        for arg in (head.args if isinstance(head, Compound) else []):
            if not isinstance(arg, Var):
                for v in sorted(vars_in(arg)):
                    if v != "_" and v not in reg_map:
                        counter += 1
                        reg_map[v] = f"X{counter}"
        return reg_map, 0

    # Permanent = appears in 2+ chunks (head=0, goal_i=i+1), with max chunk > 0
    chunk_presence = {}
    for v in all_vars:
        chunks = set()
        if v in head_vars:
            chunks.add(0)
        for gi, gvs in enumerate(goal_var_sets):
            if v in gvs:
                chunks.add(gi + 1)
        chunk_presence[v] = chunks

    permanent = set()
    for v, chunks in chunk_presence.items():
        if len(chunks) >= 2 and max(chunks) > 0:
            if chunks == {0, 1}:
                is_bare_head = (
                    isinstance(head, Compound) and
                    any(isinstance(a, Var) and a.name == v for a in head.args)
                )
                if is_bare_head:
                    continue
            permanent.add(v)

    perm_list = sorted(permanent)
    num_perm = len(perm_list)

    reg_map = {}
    for i, v in enumerate(perm_list):
        reg_map[v] = f"Y{i + 1}"

    head_arity = len(head.args) if isinstance(head, Compound) else 0
    counter = head_arity

    if isinstance(head, Compound):
        for i, arg in enumerate(head.args):
            if isinstance(arg, Var) and arg.name != "_" and arg.name not in reg_map:
                reg_map[arg.name] = f"X{i + 1}"
        for arg in head.args:
            if not isinstance(arg, Var):
                for v in sorted(vars_in(arg)):
                    if v != "_" and v not in reg_map:
                        counter += 1
                        reg_map[v] = f"X{counter}"

    for v in sorted(all_vars):
        if v != "_" and v not in reg_map:
            counter += 1
            reg_map[v] = f"X{counter}"

    return reg_map, num_perm


# ---------------------------------------------------------------------------
# Clause compiler  (emits L2 nodes directly)
# ---------------------------------------------------------------------------

class ClauseCompiler:
    """Compiles a single normalized clause to a list of L2 instruction nodes."""

    def __init__(self, head, body_goals, reg_map, num_perm):
        self.head = head
        self.body_goals = body_goals
        self.reg_map = reg_map
        self.num_perm = num_perm
        self.instrs = []
        self.seen = set()

    def compile(self) -> list:
        if self.body_goals and self.num_perm > 0:
            self.instrs.append(L2.Allocate(n=self.num_perm))

        if isinstance(self.head, Compound):
            for i, arg in enumerate(self.head.args):
                self._emit_get(arg, i + 1)
        # Atom/0 head: nothing to match

        if self.body_goals:
            for gi, goal in enumerate(self.body_goals):
                self._emit_goal(goal, is_last=(gi == len(self.body_goals) - 1))
        else:
            self.instrs.append(L2.Proceed())

        return self.instrs

    # -- Head matching --

    def _emit_get(self, arg, ai):
        if isinstance(arg, Var):
            if arg.name == "_":
                return
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name in self.seen:
                self.instrs.append(L2.GetValue(reg=vn, ai=ai))
            else:
                self.instrs.append(L2.GetVariable(reg=vn, ai=ai))
                self.seen.add(arg.name)

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(L2.GetConstant(value=val, ai=ai))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(L2.GetList(ai=ai))
            self._emit_unify(arg.args[0])
            self._emit_unify(arg.args[1])

        elif isinstance(arg, Compound):
            self.instrs.append(
                L2.GetStructure(functor=arg.functor, arity=len(arg.args), ai=ai)
            )
            for sub in arg.args:
                self._emit_unify(sub)

    # -- Subterm unification --

    def _emit_unify(self, sub):
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(L2.UnifyVoid(n=1))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(L2.UnifyVoid(n=1))
            elif sub.name not in self.seen:
                self.instrs.append(L2.UnifyVariable(reg=vn))
                self.seen.add(sub.name)
            else:
                self.instrs.append(L2.UnifyValue(reg=vn))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(L2.UnifyConstant(value=val))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            temp = self._alloc_temp()
            self.instrs.append(L2.UnifyVariable(reg=temp))
            self._deferred_gets.append((sub, temp))

        elif isinstance(sub, Compound):
            temp = self._alloc_temp()
            self.instrs.append(L2.UnifyVariable(reg=temp))
            self._deferred_gets.append((sub, temp))

    _deferred_gets = []
    _temp_counter = 100

    def _alloc_temp(self):
        ClauseCompiler._temp_counter += 1
        return f"X{ClauseCompiler._temp_counter}"

    # -- Body goals --

    def _emit_goal(self, goal, is_last):
        if isinstance(goal, Atom) and goal.name == "!":
            self.instrs.append(L2.NeckCut())
            return

        if isinstance(goal, Compound):
            for i, arg in enumerate(goal.args):
                self._emit_put(arg, i + 1)
            functor, arity = goal.functor, len(goal.args)
        elif isinstance(goal, Atom):
            functor, arity = goal.name, 0
        else:
            return

        if is_last:
            if self.num_perm > 0:
                self.instrs.append(L2.Deallocate())
            self.instrs.append(L2.Execute(functor=functor, arity=arity))
        else:
            self.instrs.append(L2.Call(functor=functor, arity=arity))

    # -- Body argument building --

    def _emit_put(self, arg, ai):
        if isinstance(arg, Var):
            if arg.name == "_":
                self.instrs.append(L2.PutVariable(reg=f"X{ai}", ai=ai))
                return
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name not in self.seen:
                self.seen.add(arg.name)
                self.instrs.append(L2.PutVariable(reg=vn, ai=ai))
            else:
                self.instrs.append(L2.PutValue(reg=vn, ai=ai))

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(L2.PutConstant(value=val, ai=ai))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(L2.PutList(ai=ai))
            self._emit_set(arg.args[0])
            self._emit_set(arg.args[1])

        elif isinstance(arg, Compound):
            self.instrs.append(
                L2.PutStructure(functor=arg.functor, arity=len(arg.args), ai=ai)
            )
            for sub in arg.args:
                self._emit_set(sub)

    def _emit_set(self, sub):
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(L2.SetVoid(n=1))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(L2.SetVoid(n=1))
            elif sub.name not in self.seen:
                self.seen.add(sub.name)
                self.instrs.append(L2.SetVariable(reg=vn))
            else:
                self.instrs.append(L2.SetValue(reg=vn))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(L2.SetConstant(value=val))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            self._emit_set(sub.args[0])
            self._emit_set(sub.args[1])

        else:
            self.instrs.append(L2.SetVoid(n=1))


# ---------------------------------------------------------------------------
# Query compiler  (emits L2 nodes directly)
# ---------------------------------------------------------------------------

class QueryCompiler:
    """Compiles a query (?- goal1, goal2, ...) to L2 instruction nodes."""

    def __init__(self, goals, reg_map):
        self.goals = goals
        self.reg_map = reg_map
        self.instrs = []
        self.seen = set()

    def compile(self) -> list:
        for gi, goal in enumerate(self.goals):
            is_last = (gi == len(self.goals) - 1)

            if isinstance(goal, Compound):
                for i, arg in enumerate(goal.args):
                    self._emit_put(arg, i + 1)
                functor, arity = goal.functor, len(goal.args)
            elif isinstance(goal, Atom):
                functor, arity = goal.name, 0
            else:
                continue

            if is_last:
                self.instrs.append(L2.Execute(functor=functor, arity=arity))
            else:
                self.instrs.append(L2.Call(functor=functor, arity=arity))

        return self.instrs

    def _emit_put(self, arg, ai):
        if isinstance(arg, Var):
            if arg.name == "_":
                self.instrs.append(L2.PutVariable(reg=f"X{ai}", ai=ai))
                return
            vn = self.reg_map.get(arg.name, f"X{ai}")
            if arg.name not in self.seen:
                self.seen.add(arg.name)
                self.instrs.append(L2.PutVariable(reg=vn, ai=ai))
            else:
                self.instrs.append(L2.PutValue(reg=vn, ai=ai))

        elif isinstance(arg, (Atom, Number)):
            val = arg.name if isinstance(arg, Atom) else arg.value
            self.instrs.append(L2.PutConstant(value=val, ai=ai))

        elif isinstance(arg, Compound) and arg.functor == "." and len(arg.args) == 2:
            self.instrs.append(L2.PutList(ai=ai))
            self._emit_set(arg.args[0])
            self._emit_set(arg.args[1])

        elif isinstance(arg, Compound):
            self.instrs.append(
                L2.PutStructure(functor=arg.functor, arity=len(arg.args), ai=ai)
            )
            for sub in arg.args:
                self._emit_set(sub)

    def _emit_set(self, sub):
        if isinstance(sub, Var):
            if sub.name == "_":
                self.instrs.append(L2.SetVoid(n=1))
                return
            vn = self.reg_map.get(sub.name)
            if vn is None:
                self.instrs.append(L2.SetVoid(n=1))
            elif sub.name not in self.seen:
                self.seen.add(sub.name)
                self.instrs.append(L2.SetVariable(reg=vn))
            else:
                self.instrs.append(L2.SetValue(reg=vn))

        elif isinstance(sub, (Atom, Number)):
            val = sub.name if isinstance(sub, Atom) else sub.value
            self.instrs.append(L2.SetConstant(value=val))

        elif isinstance(sub, Compound) and sub.functor == "." and len(sub.args) == 2:
            self._emit_set(sub.args[0])
            self._emit_set(sub.args[1])

        else:
            self.instrs.append(L2.SetVoid(n=1))


# ---------------------------------------------------------------------------
# Predicate compilation
# ---------------------------------------------------------------------------

def compile_predicate(name, arity, clauses) -> 'L2.Predicate2':
    """Compile all clauses for a predicate into a Predicate2 node."""
    compiled_clauses = []

    for ci, (head, body_goals) in enumerate(clauses):
        reg_map, num_perm = allocate_registers(head, body_goals)
        cc = ClauseCompiler(head, body_goals, reg_map, num_perm)
        clause_instrs = cc.compile()

        label = f"{name}/{arity}" if len(clauses) == 1 else f"{name}/{arity}_c{ci}"

        if len(clauses) > 1:
            if ci == 0:
                clause_instrs.insert(
                    0, L2.TryMeElse(next_label=f"{name}/{arity}_c{ci+1}", arity=arity)
                )
            elif ci < len(clauses) - 1:
                clause_instrs.insert(
                    0, L2.RetryMeElse(next_label=f"{name}/{arity}_c{ci+1}", arity=arity)
                )
            else:
                clause_instrs.insert(0, L2.TrustMe(arity=arity))

        compiled_clauses.append(L2.Clause2(label=label, instrs=clause_instrs))

    return L2.Predicate2(name=name, arity=arity, clauses=compiled_clauses)


# ---------------------------------------------------------------------------
# Program compilation
# ---------------------------------------------------------------------------

def compile_program_l2(program: Program) -> 'L2.Program2':
    """Compile a normalized L1 Program to a Program2 (L2 typed WAM nodes)."""
    pred_clauses = {}   # (name, arity) -> [(head, body_goals)]
    query_list = []

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
            all_vars = set()
            for g in clause.goals:
                collect_vars(g, all_vars)
            all_vars.discard("_")

            max_arity = max(
                (len(g.args) for g in clause.goals if isinstance(g, Compound)),
                default=0,
            )
            reg_map = {v: f"X{max_arity + i + 1}" for i, v in enumerate(sorted(all_vars))}

            qc = QueryCompiler(clause.goals, reg_map)
            query_list.append(L2.Query2(instrs=qc.compile(), reg_map=reg_map))

    pred_list = [
        compile_predicate(name, arity, clauses)
        for (name, arity), clauses in pred_clauses.items()
    ]

    return L2.Program2(predicates=pred_list, queries=query_list)
