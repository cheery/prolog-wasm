"""Prolog → LP Form lowering pass (Phase 8).

Input:  a normalized L1 Program (Atom | Number | Var | Compound only).
Output: an LPProgram that, when linked with wam_runtime.lp and compiled
        via lp_pipeline.lp_compile, implements the Prolog program.

Strategy
--------
For each Prolog predicate ``p/N`` with K clauses we emit:

* One LPProc ``p`` (the "driver") that checks FAIL at entry and, if
  clear, hands off to the per-clause try-chain.
* For multi-clause predicates, two helper procs ``p__try`` (dispatch on
  integer clause id) and ``p__retry`` (check FAIL and try the next
  clause if set).
* One LPProc ``p__cK`` per clause, whose body does head matching
  followed by body-goal execution.

For body goals we emit a chain of continuation procs. Each non-tail
goal lives in the current proc; the remainder of the body (with a
FAIL-check entry) becomes another proc that is tail-called.

The final query becomes a proc ``__query`` and the emitted module
exports ``run_get(n; val)`` that inits, runs the query, then returns
``heap_get_val(deref(XREG[n]))``.

Prolog variables are represented as heap addresses (integers flowing
through LP proc inputs / locals). A first occurrence of a variable
binds it to a freshly allocated REF cell; subsequent occurrences
reuse that address.

Only the features exercised by ``test_phase6.py`` are supported:
* single-argument facts and two-argument facts with atoms/numbers
* multi-clause predicates with ground constants (retry on FAIL)
* rules whose body is a chain of compound goals
* neck cut (``!``) — treated as a no-op under the simple retry model
  because the tests don't require post-commit backtracking

Structures, lists, and general cuts are intentionally out of scope
for this first cut; extending them is straightforward once needed.
"""

from dataclasses import dataclass

from prolog_parser import Atom, Number, Var, Compound, Fact, Rule, Query, Program
from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst,
)
from symbols import SymbolTable


# ---------------------------------------------------------------------------
# Program → LPProgram
# ---------------------------------------------------------------------------

class PrologToLP:
    """Compile a normalized Prolog Program into an LPProgram."""

    def __init__(self):
        self.syms = SymbolTable()
        self.procs: list[LPProc] = []
        self._kont_counter = 0
        # Map (predicate_name, arity) -> LP procedure name (always the same
        # as predicate_name for now; kept explicit in case we later need
        # arity-mangled names).
        self._pred_lp_name: dict[tuple[str, int], str] = {}

    # -- public --

    def compile(self, program_l1: Program) -> tuple[LPProgram, SymbolTable]:
        preds: dict[tuple[str, int], list] = {}
        query = None

        for clause in program_l1.clauses:
            if isinstance(clause, Fact):
                key = self._head_key(clause.head)
                preds.setdefault(key, []).append((clause.head, []))
            elif isinstance(clause, Rule):
                key = self._head_key(clause.head)
                preds.setdefault(key, []).append((clause.head, clause.body))
            elif isinstance(clause, Query):
                if query is None:
                    query = clause.goals
                # Additional queries are ignored — matches the behaviour
                # of the existing WAM pipeline.

        # Pre-register predicate names (so body goals can resolve them even
        # if the clause defining the predicate appears later).
        for (name, arity) in preds:
            self.syms.intern(name)
            self._pred_lp_name[(name, arity)] = self._lp_pred_name(name, arity)

        # Emit each predicate.
        for (name, arity), clauses in preds.items():
            self._emit_predicate(name, arity, clauses)

        if query is not None:
            self._emit_query(query)

        return LPProgram(procedures=list(self.procs)), self.syms

    # ------------------------------------------------------------------
    # Predicate emission
    # ------------------------------------------------------------------

    def _emit_predicate(self, name: str, arity: int, clauses: list):
        lp_name = self._pred_lp_name[(name, arity)]
        input_names = [f"a{i+1}" for i in range(arity)]

        if len(clauses) == 1:
            # Single-clause predicate: driver is a FAIL-guarded wrapper that
            # simply tail-calls the clause body.
            body_name = f"{lp_name}__c0"
            self._emit_clause(body_name, arity, clauses[0])
            self._emit_fail_guarded_wrapper(lp_name, arity, body_name)
            return

        # Multi-clause predicate: driver → __try (dispatch) → per-clause bodies
        try_name = f"{lp_name}__try"
        retry_name = f"{lp_name}__retry"

        # Driver: check FAIL at entry, then call __try(a1, ..., 0).
        self._emit_multi_driver(lp_name, arity, try_name)

        # Emit each clause body.
        for ci, clause in enumerate(clauses):
            body_name = f"{lp_name}__c{ci}"
            self._emit_clause(body_name, arity, clause)

        # __try: dispatch on integer clause id.
        self._emit_try_dispatch(try_name, retry_name, lp_name, arity, len(clauses))

        # __retry: if FAIL set, clear it and advance cid.
        self._emit_retry(retry_name, try_name, arity)

    def _emit_fail_guarded_wrapper(self, lp_name: str, arity: int, body_name: str):
        input_names = [f"a{i+1}" for i in range(arity)]

        # Clause 0: FAIL != 0, do nothing.
        head0 = LPHead(name=lp_name, inputs=input_names, outputs=[])
        goals0 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("ne", LPVar("__f"), LPConst(0)),
        ]
        clause0 = LPClause(head=head0, goals=goals0)

        # Clause 1: FAIL == 0, tail-call body.
        head1 = LPHead(name=lp_name, inputs=input_names, outputs=[])
        goals1 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("eq", LPVar("__f"), LPConst(0)),
            Call(name=body_name,
                 inputs=[LPVar(n) for n in input_names],
                 outputs=[]),
        ]
        clause1 = LPClause(head=head1, goals=goals1)

        self.procs.append(LPProc(
            name=lp_name, arity_in=arity, arity_out=0,
            clauses=[clause0, clause1],
        ))

    def _emit_multi_driver(self, lp_name: str, arity: int, try_name: str):
        input_names = [f"a{i+1}" for i in range(arity)]

        head0 = LPHead(name=lp_name, inputs=input_names, outputs=[])
        goals0 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("ne", LPVar("__f"), LPConst(0)),
        ]
        clause0 = LPClause(head=head0, goals=goals0)

        head1 = LPHead(name=lp_name, inputs=input_names, outputs=[])
        goals1 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("eq", LPVar("__f"), LPConst(0)),
            Call(name=try_name,
                 inputs=[LPVar(n) for n in input_names] + [LPConst(0)],
                 outputs=[]),
        ]
        clause1 = LPClause(head=head1, goals=goals1)

        self.procs.append(LPProc(
            name=lp_name, arity_in=arity, arity_out=0,
            clauses=[clause0, clause1],
        ))

    def _emit_try_dispatch(self, try_name: str, retry_name: str,
                           lp_name: str, arity: int, n_clauses: int):
        input_names = [f"a{i+1}" for i in range(arity)] + ["cid"]
        clauses = []

        for ci in range(n_clauses):
            body_name = f"{lp_name}__c{ci}"
            head = LPHead(name=try_name, inputs=input_names, outputs=[])
            goals = [
                Guard("eq", LPVar("cid"), LPConst(ci)),
                Call(name=body_name,
                     inputs=[LPVar(f"a{i+1}") for i in range(arity)],
                     outputs=[]),
                Call(name=retry_name,
                     inputs=[LPVar(f"a{i+1}") for i in range(arity)]
                            + [LPConst(ci)],
                     outputs=[]),
            ]
            clauses.append(LPClause(head=head, goals=goals))

        # Fallback clause: out of alternatives, set FAIL = 1.
        fallback_head = LPHead(name=try_name, inputs=input_names, outputs=[])
        fallback_goals = [
            PrimOp("gset", [LPVar("FAIL"), LPConst(1)], []),
        ]
        clauses.append(LPClause(head=fallback_head, goals=fallback_goals))

        self.procs.append(LPProc(
            name=try_name, arity_in=len(input_names), arity_out=0,
            clauses=clauses,
        ))

    def _emit_retry(self, retry_name: str, try_name: str, arity: int):
        input_names = [f"a{i+1}" for i in range(arity)] + ["cid"]

        head0 = LPHead(name=retry_name, inputs=input_names, outputs=[])
        goals0 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("eq", LPVar("__f"), LPConst(0)),
        ]
        clause0 = LPClause(head=head0, goals=goals0)

        head1 = LPHead(name=retry_name, inputs=input_names, outputs=[])
        goals1 = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("ne", LPVar("__f"), LPConst(0)),
            PrimOp("gset", [LPVar("FAIL"), LPConst(0)], []),
            PrimOp("add", [LPVar("cid"), LPConst(1)], ["nc"]),
            Call(name=try_name,
                 inputs=[LPVar(f"a{i+1}") for i in range(arity)] + [LPVar("nc")],
                 outputs=[]),
        ]
        clause1 = LPClause(head=head1, goals=goals1)

        self.procs.append(LPProc(
            name=retry_name, arity_in=len(input_names), arity_out=0,
            clauses=[clause0, clause1],
        ))

    # ------------------------------------------------------------------
    # Clause body emission
    # ------------------------------------------------------------------

    def _emit_clause(self, body_name: str, arity: int, clause):
        head, body_goals = clause
        input_names = [f"a{i+1}" for i in range(arity)]

        # var_env: Prolog variable name -> LP variable name holding that
        # variable's heap address.
        var_env: dict[str, str] = {}
        fresh_counter = [0]

        def fresh(prefix: str) -> str:
            fresh_counter[0] += 1
            return f"__{prefix}{fresh_counter[0]}"

        goals: list = []

        # --- Head matching ---
        if isinstance(head, Compound):
            for i, arg in enumerate(head.args):
                self._compile_head_match(
                    goals, f"a{i+1}", arg, var_env, fresh)
        # Atom head (arity 0) or Var head: nothing to match.

        # --- Body ---
        self._compile_body_chain(
            body_name, input_names, arity,
            goals, var_env, fresh, body_goals)

    def _compile_body_chain(self, proc_name: str, input_names: list[str],
                            arity: int, initial_goals: list,
                            var_env: dict, fresh, body_goals: list):
        """Emit one or more procs that together execute body_goals.

        The first proc is ``proc_name`` with the given input_names and
        seeded ``initial_goals`` (typically head-matching code).

        Each non-tail body goal gets the remaining goals placed into a
        continuation proc that checks FAIL at entry. Tail goals are
        emitted inline.
        """
        if not body_goals:
            # Fact or clause with empty body.
            self._commit_proc(proc_name, input_names, initial_goals)
            return

        current_proc = proc_name
        current_inputs = list(input_names)
        current_env = dict(var_env)
        current_goals = list(initial_goals)

        for gi, goal in enumerate(body_goals):
            is_last = (gi == len(body_goals) - 1)

            # Cut: our simplified retry model doesn't support across-call
            # backtracking, so a neck cut is a no-op.
            if isinstance(goal, Atom) and goal.name == "!":
                if is_last:
                    self._commit_proc(current_proc, current_inputs, current_goals)
                    return
                continue

            # Compile the goal: build the arg cells, then call the predicate.
            pred_name, pred_arity, arg_vars = self._compile_call_args(
                goal, current_goals, current_env, fresh)
            callee = self._pred_lp_name.get((pred_name, pred_arity))
            if callee is None:
                # Unknown predicate — register it under its own name so the
                # generated code still references *something*. This gives a
                # loud error at link/validate time rather than a silent miss.
                callee = self._lp_pred_name(pred_name, pred_arity)
                self._pred_lp_name[(pred_name, pred_arity)] = callee

            call = Call(
                name=callee,
                inputs=[LPVar(v) for v in arg_vars],
                outputs=[],
            )

            if is_last:
                current_goals.append(call)
                self._commit_proc(current_proc, current_inputs, current_goals)
                return

            # Non-tail call: emit the call, then tail-call a continuation.
            current_goals.append(call)

            # The continuation proc needs every variable still live —
            # approximate by carrying all env vars plus head inputs.
            live = self._collect_live_vars(current_env, body_goals[gi + 1:])

            kont_name = self._fresh_kont()
            kont_inputs = sorted(live)

            current_goals.append(Call(
                name=kont_name,
                inputs=[LPVar(v) for v in kont_inputs],
                outputs=[],
            ))
            self._commit_proc(current_proc, current_inputs, current_goals)

            # Seed the continuation with a FAIL check.
            continuation_initial = [
                PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            ]
            # Split into two clauses: FAIL!=0 → no-op; FAIL==0 → continue.
            # We'll emit those via a dedicated helper since _commit_proc
            # only handles single-clause procs.
            self._emit_kont_proc(
                kont_name, kont_inputs, current_env, fresh,
                body_goals[gi + 1:])
            return  # the continuation handles remaining goals

        # Fallthrough (shouldn't happen because is_last returns above).
        self._commit_proc(current_proc, current_inputs, current_goals)

    def _emit_kont_proc(self, kont_name: str, kont_inputs: list[str],
                         var_env: dict, fresh, remaining_goals: list):
        """Emit a continuation proc: FAIL→skip, otherwise execute the rest."""
        # Clause 0: FAIL != 0 — do nothing.
        head_fail = LPHead(name=kont_name, inputs=kont_inputs, outputs=[])
        goals_fail = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("ne", LPVar("__f"), LPConst(0)),
        ]
        self.procs.append(LPProc(
            name=kont_name, arity_in=len(kont_inputs), arity_out=0,
            clauses=[LPClause(head=head_fail, goals=goals_fail)],
        ))
        # Immediately augment the proc with the "ok" clause below; we'll
        # do this by compiling the remaining goals into a helper and
        # having the kont's "ok" branch tail-call it.
        ok_goals = [
            PrimOp("gget", [LPVar("FAIL")], ["__f"]),
            Guard("eq", LPVar("__f"), LPConst(0)),
        ]
        # The helper proc does the actual rest-of-body work.
        helper_name = kont_name + "__ok"
        ok_goals.append(Call(
            name=helper_name,
            inputs=[LPVar(v) for v in kont_inputs],
            outputs=[],
        ))
        # Append the "ok" clause to the just-added kont proc.
        self.procs[-1].clauses.append(LPClause(
            head=LPHead(name=kont_name, inputs=kont_inputs, outputs=[]),
            goals=ok_goals,
        ))

        # Recurse: the helper proc embodies the remaining goals.
        self._compile_body_chain(
            helper_name, kont_inputs, len(kont_inputs),
            [], var_env, fresh, remaining_goals)

    def _commit_proc(self, name: str, input_names: list[str], goals: list):
        clause = LPClause(
            head=LPHead(name=name, inputs=input_names, outputs=[]),
            goals=goals,
        )
        self.procs.append(LPProc(
            name=name, arity_in=len(input_names), arity_out=0,
            clauses=[clause],
        ))

    # ------------------------------------------------------------------
    # Head matching
    # ------------------------------------------------------------------

    def _compile_head_match(self, goals: list, arg_name: str, arg, var_env,
                            fresh):
        """Match a clause head argument `arg` against the heap cell at
        `arg_name` (an LP variable holding a heap address)."""
        if isinstance(arg, Var):
            if arg.name == "_":
                return
            if arg.name in var_env:
                # Seen before: unify the two heap addrs.
                goals.append(Call(
                    name="unify",
                    inputs=[LPVar(arg_name), LPVar(var_env[arg.name])],
                    outputs=[],
                ))
            else:
                # First occurrence: alias Prolog var to the arg address.
                # Since LP Form is single-assignment, we need a dedicated
                # local.
                bind_name = fresh(f"v_{arg.name}_")
                goals.append(PrimOp(
                    "copy", [LPVar(arg_name)], [bind_name],
                ))
                var_env[arg.name] = bind_name
            return

        if isinstance(arg, (Atom, Number)):
            val = self.syms.encode_constant(
                arg.name if isinstance(arg, Atom) else arg.value)
            goals.append(Call(
                name="match_constant",
                inputs=[LPVar(arg_name), LPConst(val)],
                outputs=[],
            ))
            return

        raise NotImplementedError(
            f"head match for {type(arg).__name__} not yet supported")

    # ------------------------------------------------------------------
    # Body call compilation
    # ------------------------------------------------------------------

    def _compile_call_args(self, goal, goals: list, var_env, fresh):
        """Build the argument cells for a body goal, appending to `goals`.

        Returns (pred_name, arity, [arg_lp_var_names]).
        """
        if isinstance(goal, Atom):
            return goal.name, 0, []

        if isinstance(goal, Compound):
            arg_vars = []
            for arg in goal.args:
                arg_vars.append(self._materialize_arg(arg, goals, var_env, fresh))
            return goal.functor, len(goal.args), arg_vars

        raise NotImplementedError(
            f"body goal {type(goal).__name__} not supported")

    def _materialize_arg(self, arg, goals: list, var_env, fresh) -> str:
        """Materialize a Prolog term as a heap address (LP variable).

        Returns the LP variable name holding the heap address.
        """
        if isinstance(arg, Var):
            if arg.name == "_":
                # Fresh anonymous var.
                addr = fresh("anon_")
                goals.append(Call(
                    name="put_variable",
                    inputs=[],
                    outputs=[addr]))
                return addr
            if arg.name in var_env:
                return var_env[arg.name]
            # First body occurrence of a new variable: allocate a fresh REF
            # cell and record it in the env.
            addr = fresh(f"v_{arg.name}_")
            goals.append(Call(
                name="put_variable", inputs=[], outputs=[addr],
            ))
            var_env[arg.name] = addr
            return addr

        if isinstance(arg, (Atom, Number)):
            val = self.syms.encode_constant(
                arg.name if isinstance(arg, Atom) else arg.value)
            addr = fresh("c_")
            goals.append(Call(
                name="put_constant",
                inputs=[LPConst(val)],
                outputs=[addr],
            ))
            return addr

        raise NotImplementedError(
            f"materialize {type(arg).__name__} not yet supported")

    # ------------------------------------------------------------------
    # Query + run_get entry point
    # ------------------------------------------------------------------

    def _emit_query(self, goals: list):
        """Emit __query and run_get procs for the first query.

        Strategy: allocate query variables as REF cells, store their
        addresses into XREG (both argument positions and primary slots),
        then call the predicate.  ``run_get`` inits the heap, runs
        __query, and returns heap_get_val(deref(XREG[n])).
        """
        all_vars = []
        seen = set()
        for g in goals:
            self._collect_vars(g, all_vars, seen)

        max_arity = max(
            (len(g.args) for g in goals if isinstance(g, Compound)),
            default=0,
        )
        var_slot: dict[str, int] = {}
        for i, v in enumerate(sorted(all_vars)):
            var_slot[v] = max_arity + i + 1

        query_goals: list = []
        var_env: dict[str, str] = {}
        fresh_counter = [0]

        def fresh(prefix: str) -> str:
            fresh_counter[0] += 1
            return f"__q{fresh_counter[0]}_{prefix}"

        # Build args for each goal, store into XREG argument positions,
        # then call the predicate.
        for goal in goals:
            if isinstance(goal, Atom):
                pred_name, pred_arity = goal.name, 0
                arg_addrs = []
            elif isinstance(goal, Compound):
                pred_name = goal.functor
                pred_arity = len(goal.args)
                arg_addrs = []
                for arg in goal.args:
                    arg_addrs.append(
                        self._materialize_arg(arg, query_goals, var_env, fresh))
            else:
                raise NotImplementedError(f"query goal: {type(goal)}")

            # Store argument addresses into XREG[1..arity]
            for ai, addr in enumerate(arg_addrs):
                query_goals.append(PrimOp(
                    "aset", [LPVar("XREG"), LPConst(ai + 1), LPVar(addr)],
                    [],
                ))

            # Call the predicate
            callee = self._pred_lp_name.get((pred_name, pred_arity))
            if callee is None:
                callee = self._lp_pred_name(pred_name, pred_arity)
                self._pred_lp_name[(pred_name, pred_arity)] = callee

            query_goals.append(Call(
                name=callee,
                inputs=[LPVar(a) for a in arg_addrs],
                outputs=[],
            ))

        # Store query variables into their primary XREG slots
        for v in sorted(all_vars):
            if v in var_env and v in var_slot:
                query_goals.append(PrimOp(
                    "aset", [LPVar("XREG"), LPConst(var_slot[v]),
                             LPVar(var_env[v])],
                    [],
                ))

        self._commit_proc("__query", [], query_goals)

        # run_get(n; val): init + __query + deref(XREG[n]) + heap_get_val
        rg_goals = [
            Call(name="init", inputs=[], outputs=[]),
            Call(name="__query", inputs=[], outputs=[]),
            PrimOp("aget", [LPVar("XREG"), LPVar("n")], ["addr"]),
            Call(name="deref", inputs=[LPVar("addr")], outputs=["d"]),
            Call(name="heap_get_val", inputs=[LPVar("d")], outputs=["val"]),
        ]
        self.procs.append(LPProc(
            name="run_get", arity_in=1, arity_out=1,
            clauses=[LPClause(
                head=LPHead(name="run_get", inputs=["n"], outputs=["val"]),
                goals=rg_goals,
            )],
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _head_key(self, head):
        if isinstance(head, Compound):
            return (head.functor, len(head.args))
        if isinstance(head, Atom):
            return (head.name, 0)
        raise ValueError(f"bad clause head: {head!r}")

    def _lp_pred_name(self, name: str, arity: int) -> str:
        # Sanitize: Prolog atom names may contain characters LP Form's
        # lexer rejects. We only map '.' for now; extend if needed.
        safe = name.replace(".", "_dot_").replace("'", "_q_")
        return f"pl_{safe}_{arity}"

    def _fresh_kont(self) -> str:
        self._kont_counter += 1
        return f"__kont{self._kont_counter}"

    def _collect_vars(self, term, out: list, seen: set):
        if isinstance(term, Var):
            if term.name != "_" and term.name not in seen:
                seen.add(term.name)
                out.append(term.name)
        elif isinstance(term, Compound):
            for a in term.args:
                self._collect_vars(a, out, seen)

    def _collect_live_vars(self, var_env: dict, remaining_goals: list) -> set:
        """Approximate live-variable set: every Prolog var that appears in
        the remaining goals and is bound in var_env."""
        referenced = set()
        seen = set()
        for g in remaining_goals:
            vars_in_goal = []
            self._collect_vars(g, vars_in_goal, seen)
            referenced.update(vars_in_goal)
        live = set()
        for vname in referenced:
            lp = var_env.get(vname)
            if lp is not None:
                live.add(lp)
        return live


def compile_prolog(program_l1: Program) -> tuple[LPProgram, SymbolTable]:
    """Compile a normalized L1 Prolog program to an LPProgram.

    Returns (lp_program, symbol_table).
    """
    return PrologToLP().compile(program_l1)
