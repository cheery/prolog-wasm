"""Host-side trace tooling for LP Form programs.

Three layers:

1. `Trace` / `TraceRecord` — decode the flat i32 cells produced by the
   trace-instrumented emitter (`lp_emit.py:538-593`) into structured
   records; provide indexing, iteration, filtering, and pretty printing.

2. `Replay` — a pure-Python LP Form interpreter driven by the trace.
   Each trace record identifies a clause firing; the interpreter pushes
   a frame, executes body PrimOps, and on a Call stops so the next
   record drives the callee. Guards are no-ops: the trace is the proof
   they held. Supports forward `step()`, `goto(N)` with snapshot-backed
   reverse, and `back()`.

3. A `cmd.Cmd`-based CLI (`DebuggerCLI`) and `__main__` entry that
   compiles an LP file with tracing, runs it via wasmtime, and opens an
   interactive debugger prompt.
"""

from __future__ import annotations

import cmd
import copy
import re
from dataclasses import dataclass, field

from lp_form import (
    LPProgram, LPProc, LPClause, LPHead,
    PrimOp, Guard, Call, LPVar, LPConst,
    _fmt_goal,
)


# ---------------------------------------------------------------------------
# Layer A — Trace decoder
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    proc_id: int
    clause_idx: int
    inputs: list          # list[int]
    cell_offset: int      # offset in the raw buffer where this record starts


class Trace:
    """A decoded trace.

    Construct from raw i32 cells (as returned by iterating `__trace_get`)
    and optionally the `LPProgram` that produced them so proc ids can be
    resolved back to names in pretty output.
    """

    def __init__(self, cells, program: LPProgram | None = None):
        self._records = self._decode(cells)
        self.program = program

    @staticmethod
    def _decode(cells):
        records = []
        i = 0
        n = len(cells)
        while i < n:
            size = cells[i]
            proc_id = cells[i + 1]
            clause_idx = cells[i + 2]
            inputs = list(cells[i + 3 : i + 1 + size])
            records.append(TraceRecord(
                proc_id=proc_id,
                clause_idx=clause_idx,
                inputs=inputs,
                cell_offset=i,
            ))
            i += 1 + size
        return records

    @classmethod
    def from_module(cls, tm, program: LPProgram | None = None) -> "Trace":
        """Construct from a host module exposing `__trace_len`/`__trace_get`."""
        n = tm.call("__trace_len")
        cells = [tm.call("__trace_get", i) for i in range(n)]
        return cls(cells, program)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, i):
        return self._records[i]

    def __iter__(self):
        return iter(self._records)

    def proc_name(self, proc_id: int) -> str:
        if self.program is not None and 0 <= proc_id < len(self.program.procedures):
            return self.program.procedures[proc_id].name
        return f"<proc{proc_id}>"

    def filter(self, proc_name: str) -> list:
        if self.program is None:
            raise ValueError("Trace.filter requires a program for name lookup")
        pid = None
        for i, p in enumerate(self.program.procedures):
            if p.name == proc_name:
                pid = i
                break
        if pid is None:
            raise KeyError(f"no such procedure: {proc_name}")
        return [r for r in self._records if r.proc_id == pid]

    def pretty(self, i: int) -> str:
        r = self._records[i]
        name = self.proc_name(r.proc_id)
        args = ", ".join(str(x) for x in r.inputs)
        return f"[{i}] {name}/{r.clause_idx}({args})"

    def dump(self, start: int = 0, end: int | None = None) -> str:
        if end is None:
            end = len(self._records)
        return "\n".join(self.pretty(i) for i in range(start, end))


# ---------------------------------------------------------------------------
# Layer B — Replay interpreter
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    proc_name: str
    clause_idx: int
    goal_idx: int
    var_map: dict          # str -> int


_MUTATING_PRIMOPS = {"gset", "aset", "anew", "rnew"}


class Replay:
    """Pure-Python LP Form interpreter driven by a Trace.

    State shape:
      - globals: dict[str, int]
      - arrays:  dict[str, list[int]]  (populated on anew)
      - refs:    dict[str, list]        (populated on rnew; rget/rset unused)
      - frames:  list[Frame]            (root frame at index 0)
      - cursor:  int (next trace record to consume)
      - final_outputs: dict | None (set when the root frame returns)
    """

    def __init__(self, program: LPProgram, trace: Trace,
                 snapshot_interval: int = 0):
        self.program = program
        self.trace = trace
        self.proc_by_id = {i: p for i, p in enumerate(program.procedures)}
        self.proc_by_name = {p.name: p for p in program.procedures}
        self._snapshot_interval = snapshot_interval
        self.reset()

    # ---- lifecycle ---------------------------------------------------

    def reset(self):
        self.globals = {g.name: g.initial for g in self.program.globals}
        self.arrays = {}
        self.refs = {}
        self.frames = []
        self.cursor = 0
        self.final_outputs = None
        self._snapshots = {0: self._snapshot()}

    # ---- stepping ----------------------------------------------------

    @staticmethod
    def _preamble_split(clause):
        """Split clause goals into preamble (up to last Guard) and suffix.

        Mirrors the emitter's split in ``_emit_clause_chain``: preamble
        goals execute before the trace record is written; suffix goals
        execute after.  The Replay must skip the preamble (already
        accounted for via earlier trace records) and re-execute it
        inline only to recover intermediate variable values needed by
        the suffix.
        """
        last_guard = -1
        for i, g in enumerate(clause.goals):
            if isinstance(g, Guard):
                last_guard = i
        if last_guard < 0:
            return [], clause.goals
        return clause.goals[:last_guard + 1], clause.goals[last_guard + 1:]

    def step(self) -> TraceRecord | None:
        """Consume one trace record; return it, or None if trace exhausted."""
        if self.cursor >= len(self.trace):
            return None
        rec = self.trace[self.cursor]
        self.cursor += 1
        proc = self.proc_by_id[rec.proc_id]
        clause = proc.clauses[rec.clause_idx]
        var_map = {}
        for name, val in zip(clause.head.inputs, rec.inputs):
            var_map[name] = val

        # Multi-clause procs: the emitter executes preamble goals
        # (Calls before the last Guard) *before* writing the trace
        # record, so those sub-call records appear earlier in the
        # trace.  Re-execute the preamble inline to recover any
        # intermediate values the suffix needs.
        start_idx = 0
        if len(proc.clauses) > 1:
            preamble, suffix = self._preamble_split(clause)
            if preamble:
                self._exec_goals_inline(preamble, var_map)
                start_idx = len(preamble)

        self.frames.append(Frame(
            proc_name=proc.name,
            clause_idx=rec.clause_idx,
            goal_idx=start_idx,
            var_map=var_map,
        ))
        self._run_to_pause()
        if (self._snapshot_interval > 0
                and self.cursor % self._snapshot_interval == 0
                and self.cursor not in self._snapshots):
            self._snapshots[self.cursor] = self._snapshot()
        return rec

    def next(self) -> TraceRecord | None:
        """Advance until the current top frame advances past its current goal
        (or returns). Useful for "step over" a Call."""
        if not self.frames:
            return self.step()
        target = self.frames[-1]
        saved_id = id(target)
        saved_depth = len(self.frames)
        saved_goal = target.goal_idx
        rec = None
        while self.cursor < len(self.trace):
            rec = self.step()
            if rec is None:
                break
            # If the saved frame is no longer on the stack, it has returned.
            if len(self.frames) < saved_depth:
                break
            top = self.frames[-1] if self.frames else None
            if top is not None and id(top) == saved_id and top.goal_idx > saved_goal:
                break
            # Root returned; frames empty
            if not self.frames:
                break
        return rec

    def back(self) -> None:
        self.goto(self.cursor - 1)

    def goto(self, target: int) -> None:
        if target < 0:
            target = 0
        if target > len(self.trace):
            target = len(self.trace)
        if target < self.cursor:
            # Rewind: restore the nearest snapshot <= target, then replay.
            candidates = [k for k in self._snapshots if k <= target]
            k = max(candidates)
            self._restore(self._snapshots[k])
        while self.cursor < target:
            if self.step() is None:
                break

    def current_record(self) -> TraceRecord | None:
        if self.cursor == 0 or self.cursor > len(self.trace):
            return None
        return self.trace[self.cursor - 1]

    # ---- snapshots ---------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            "globals": dict(self.globals),
            "arrays": {k: list(v) for k, v in self.arrays.items()},
            "refs": {k: list(v) for k, v in self.refs.items()},
            "frames": copy.deepcopy(self.frames),
            "cursor": self.cursor,
            "final_outputs": (dict(self.final_outputs)
                              if self.final_outputs is not None else None),
        }

    def _restore(self, snap: dict) -> None:
        self.globals = dict(snap["globals"])
        self.arrays = {k: list(v) for k, v in snap["arrays"].items()}
        self.refs = {k: list(v) for k, v in snap["refs"].items()}
        self.frames = copy.deepcopy(snap["frames"])
        self.cursor = snap["cursor"]
        self.final_outputs = (dict(snap["final_outputs"])
                              if snap["final_outputs"] is not None else None)

    # ---- core execution ---------------------------------------------

    def _run_to_pause(self) -> None:
        """Run the top frame's goals until a non-invertible Call is reached
        or the stack unwinds past the root frame. Calls to procs marked
        invertible are executed inline (no trace record consumed) because
        the emitter suppresses their records."""
        while self.frames:
            frame = self.frames[-1]
            proc = self.proc_by_name[frame.proc_name]
            clause = proc.clauses[frame.clause_idx]

            if frame.goal_idx >= len(clause.goals):
                # End of clause: pop and plumb outputs to caller.
                outs_by_name = {n: frame.var_map[n]
                                for n in clause.head.outputs}
                callee_name = frame.proc_name
                self.frames.pop()
                if not self.frames:
                    self.final_outputs = outs_by_name
                    return
                caller = self.frames[-1]
                caller_proc = self.proc_by_name[caller.proc_name]
                caller_clause = caller_proc.clauses[caller.clause_idx]
                call_goal = caller_clause.goals[caller.goal_idx]
                assert isinstance(call_goal, Call), \
                    f"expected Call at caller goal_idx, got {call_goal}"
                # Multi-clause preamble mismatch: the emitter executes
                # preamble Calls (e.g. heap_get_tag inside deref) before
                # emitting the parent's trace record, so those records
                # arrive earlier. When the callee doesn't match the
                # expected Call, skip output mapping and don't advance.
                if call_goal.name != callee_name:
                    continue
                for cname, caller_out in zip(
                        clause.head.outputs, call_goal.outputs):
                    caller.var_map[caller_out] = outs_by_name[cname]
                caller.goal_idx += 1
                continue

            goal = clause.goals[frame.goal_idx]
            if isinstance(goal, Guard):
                frame.goal_idx += 1
            elif isinstance(goal, PrimOp):
                self._exec_primop(goal, frame.var_map)
                frame.goal_idx += 1
            elif isinstance(goal, Call):
                target = self.proc_by_name.get(goal.name)
                if target is not None and target.invertible:
                    self._exec_invertible_call(goal, frame)
                    frame.goal_idx += 1
                    continue
                # Normal Call: pause for the next record to drive the callee.
                return
            else:
                raise ValueError(f"unknown goal: {goal!r}")

    def _exec_invertible_call(self, call: Call, caller: Frame) -> None:
        target = self.proc_by_name[call.name]
        if len(target.clauses) != 1:
            raise NotImplementedError(
                f"invertible multi-clause procs not supported in replay "
                f"(proc: {target.name})")
        clause = target.clauses[0]
        inputs = [self._val(v, caller.var_map) for v in call.inputs]
        var_map = {}
        for name, val in zip(clause.head.inputs, inputs):
            var_map[name] = val
        for g in clause.goals:
            if isinstance(g, Guard):
                continue
            if isinstance(g, PrimOp):
                self._exec_primop(g, var_map)
            else:
                raise ValueError(
                    f"invertible proc {target.name} contains non-PrimOp "
                    f"goal: {g!r}")
        for cname, caller_out in zip(clause.head.outputs, call.outputs):
            caller.var_map[caller_out] = var_map[cname]

    def _exec_goals_inline(self, goals, var_map):
        """Execute a list of goals inline (no frame push, no trace record).

        Used for re-executing multi-clause preamble goals to recover
        intermediate variable values.  Guards are treated as no-ops;
        Calls are dispatched to ``_exec_call_inline``.
        """
        for g in goals:
            if isinstance(g, Guard):
                continue
            if isinstance(g, PrimOp):
                self._exec_primop(g, var_map)
            elif isinstance(g, Call):
                self._exec_call_inline(g, var_map)

    def _exec_call_inline(self, call, var_map):
        """Execute a procedure call inline, evaluating guards to find the
        matching clause.  Recursively handles sub-calls in the callee body."""
        target = self.proc_by_name[call.name]
        inputs = [self._val(v, var_map) for v in call.inputs]
        for clause in target.clauses:
            clause_vm = {}
            for name, val in zip(clause.head.inputs, inputs):
                clause_vm[name] = val
            if self._try_clause_inline(clause, clause_vm):
                for cname, caller_out in zip(clause.head.outputs, call.outputs):
                    var_map[caller_out] = clause_vm[cname]
                return
        raise ValueError(
            f"no matching clause for inline call {call.name}")

    def _try_clause_inline(self, clause, var_map):
        """Try to execute a clause inline.  Returns True if all guards pass."""
        for g in clause.goals:
            if isinstance(g, Guard):
                left = self._val(g.left, var_map)
                right = self._val(g.right, var_map)
                if not self._check_guard(g.op, left, right):
                    return False
            elif isinstance(g, PrimOp):
                self._exec_primop(g, var_map)
            elif isinstance(g, Call):
                self._exec_call_inline(g, var_map)
        return True

    @staticmethod
    def _check_guard(op, left, right):
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        raise ValueError(f"unknown guard op: {op}")

    def _val(self, v, var_map):
        if isinstance(v, LPVar):
            return var_map[v.name]
        if isinstance(v, LPConst):
            return v.value
        raise ValueError(f"unexpected value: {v!r}")

    def _exec_primop(self, op: PrimOp, var_map: dict) -> None:
        o = op.op
        if o == "copy":
            var_map[op.outputs[0]] = self._val(op.inputs[0], var_map)
        elif o == "add":
            var_map[op.outputs[0]] = (self._val(op.inputs[0], var_map)
                                      + self._val(op.inputs[1], var_map))
        elif o == "sub":
            var_map[op.outputs[0]] = (self._val(op.inputs[0], var_map)
                                      - self._val(op.inputs[1], var_map))
        elif o == "mul":
            var_map[op.outputs[0]] = (self._val(op.inputs[0], var_map)
                                      * self._val(op.inputs[1], var_map))
        elif o == "div":
            var_map[op.outputs[0]] = self._i32_div_s(
                self._val(op.inputs[0], var_map),
                self._val(op.inputs[1], var_map))
        elif o == "rem":
            var_map[op.outputs[0]] = self._i32_rem_s(
                self._val(op.inputs[0], var_map),
                self._val(op.inputs[1], var_map))
        elif o == "and":
            var_map[op.outputs[0]] = (self._val(op.inputs[0], var_map)
                                      & self._val(op.inputs[1], var_map))
        elif o == "or":
            var_map[op.outputs[0]] = (self._val(op.inputs[0], var_map)
                                      | self._val(op.inputs[1], var_map))
        elif o == "gget":
            gname = op.inputs[0].name
            var_map[op.outputs[0]] = self.globals[gname]
        elif o == "gset":
            gname = op.inputs[0].name
            self.globals[gname] = self._val(op.inputs[1], var_map)
        elif o == "aget":
            arr = op.inputs[0].name
            idx = self._val(op.inputs[1], var_map)
            var_map[op.outputs[0]] = self.arrays[arr][idx]
        elif o == "aset":
            arr = op.inputs[0].name
            idx = self._val(op.inputs[1], var_map)
            self.arrays[arr][idx] = self._val(op.inputs[2], var_map)
        elif o == "anew":
            arr = op.inputs[0].name
            size = self._val(op.inputs[1], var_map)
            self.arrays[arr] = [0] * size
        elif o == "rnew":
            arr = op.inputs[0].name
            size = self._val(op.inputs[1], var_map)
            self.refs[arr] = [None] * size
        elif o in ("rget", "rset"):
            raise NotImplementedError(
                f"replay interpreter does not support funcref op '{o}'")
        else:
            raise ValueError(f"unknown PrimOp: {o}")

    @staticmethod
    def _i32_div_s(a: int, b: int) -> int:
        """i32.div_s semantics: truncation toward zero."""
        q, r = divmod(a, b)
        if r != 0 and (r < 0) != (b < 0):
            q += 1
        return q

    @staticmethod
    def _i32_rem_s(a: int, b: int) -> int:
        return a - Replay._i32_div_s(a, b) * b

    # ---- inspection --------------------------------------------------

    _SLICE_RE = re.compile(r"^(\w+)\[([^\]]+)\]$")

    def resolve(self, expr: str):
        """Evaluate a print expression.

        Grammar:
            stack          — formatted call-stack dump
            record         — the current trace record
            NAME           — global, or local var in current frame
            NAME[i]        — i32 array element (or ref array element)
            NAME[a:b]      — slice of an array
        """
        expr = expr.strip()
        if expr == "stack":
            return self._fmt_stack()
        if expr == "record":
            r = self.current_record()
            if r is None:
                return "(no record)"
            return self.trace.pretty(self.cursor - 1)

        m = self._SLICE_RE.match(expr)
        if m:
            name = m.group(1)
            sub = m.group(2).strip()
            container = None
            if name in self.arrays:
                container = self.arrays[name]
            elif name in self.refs:
                container = self.refs[name]
            else:
                raise KeyError(f"no array/ref named {name!r}")
            if ":" in sub:
                parts = sub.split(":")
                a = int(parts[0]) if parts[0].strip() else None
                b = int(parts[1]) if parts[1].strip() else None
                return container[a:b]
            return container[int(sub)]

        if expr in self.globals:
            return self.globals[expr]
        if self.frames and expr in self.frames[-1].var_map:
            return self.frames[-1].var_map[expr]
        if expr in self.arrays:
            return f"<array {expr}, len={len(self.arrays[expr])}>"
        if self.final_outputs and expr in self.final_outputs:
            return self.final_outputs[expr]
        raise KeyError(f"no binding for {expr!r}")

    def _fmt_stack(self) -> str:
        if not self.frames:
            return "(empty stack)"
        lines = []
        for i, f in enumerate(self.frames):
            proc = self.proc_by_name[f.proc_name]
            clause = proc.clauses[f.clause_idx]
            lines.append(
                f"  #{i} {f.proc_name}/{f.clause_idx} "
                f"@ goal {f.goal_idx}/{len(clause.goals)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer C — Interactive CLI
# ---------------------------------------------------------------------------

class DebuggerCLI(cmd.Cmd):
    intro = "LP Form trace debugger. Type 'help' or '?' for commands."
    prompt = "(lpdb) "

    def __init__(self, replay: Replay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay = replay
        self.breakpoints: set = set()

    def _show_where(self):
        r = self.replay.current_record()
        if r is None:
            self.stdout.write(
                f"at start (cursor={self.replay.cursor})\n")
        else:
            self.stdout.write(
                self.replay.trace.pretty(self.replay.cursor - 1) + "\n")

    def do_step(self, arg):
        """step — advance one trace record."""
        r = self.replay.step()
        if r is None:
            self.stdout.write("end of trace\n")
        else:
            self._show_where()
    do_s = do_step

    def do_next(self, arg):
        """next — advance until the current frame advances or returns."""
        self.replay.next()
        self._show_where()
    do_n = do_next

    def do_back(self, arg):
        """back — step backward by one record."""
        self.replay.back()
        self._show_where()
    do_b = do_back

    def do_goto(self, arg):
        """goto N — jump to record index N."""
        try:
            n = int(arg.strip())
        except ValueError:
            self.stdout.write("usage: goto N\n")
            return
        self.replay.goto(n)
        self._show_where()
    do_g = do_goto

    def do_print(self, arg):
        """print EXPR — inspect a global, local, array cell/slice, 'stack', or 'record'."""
        try:
            result = self.replay.resolve(arg.strip())
        except (KeyError, IndexError, ValueError) as e:
            self.stdout.write(f"error: {e}\n")
            return
        self.stdout.write(f"{result}\n")
    do_p = do_print

    def do_list(self, arg):
        """list — show the current clause with '->' on the current goal."""
        if not self.replay.frames:
            self.stdout.write("(no current frame)\n")
            return
        frame = self.replay.frames[-1]
        proc = self.replay.proc_by_name[frame.proc_name]
        clause = proc.clauses[frame.clause_idx]
        self.stdout.write(f"{frame.proc_name}/{frame.clause_idx}:\n")
        for i, g in enumerate(clause.goals):
            marker = "->" if i == frame.goal_idx else "  "
            self.stdout.write(f"  {marker} {i}: {_fmt_goal(g)}\n")
        if frame.goal_idx >= len(clause.goals):
            self.stdout.write("  -> (end of clause)\n")
    do_l = do_list

    def do_where(self, arg):
        """where — show call stack."""
        self.stdout.write(self.replay._fmt_stack() + "\n")
    do_w = do_where

    def do_trace(self, arg):
        """trace — pretty-print the current trace record."""
        r = self.replay.current_record()
        if r is None:
            self.stdout.write(
                f"cursor at {self.replay.cursor}; no record consumed\n")
        else:
            self.stdout.write(
                self.replay.trace.pretty(self.replay.cursor - 1) + "\n")
    do_t = do_trace

    def do_break(self, arg):
        """break PROC — add a procedure name to the breakpoint set."""
        name = arg.strip()
        if not name:
            self.stdout.write(
                f"breakpoints: {sorted(self.breakpoints)}\n")
            return
        self.breakpoints.add(name)
        self.stdout.write(f"breakpoints: {sorted(self.breakpoints)}\n")

    def do_continue(self, arg):
        """continue — step forward until a breakpoint is hit or trace ends."""
        while self.replay.cursor < len(self.replay.trace):
            r = self.replay.step()
            if r is None:
                break
            pname = self.replay.proc_by_id[r.proc_id].name
            if pname in self.breakpoints:
                self.stdout.write(
                    "break: "
                    + self.replay.trace.pretty(self.replay.cursor - 1)
                    + "\n")
                return
        self.stdout.write("end of trace\n")
    do_c = do_continue

    def do_reset(self, arg):
        """reset — rewind to the start of the trace."""
        self.replay.reset()
        self.stdout.write("reset\n")

    def do_quit(self, arg):
        """quit — exit the debugger."""
        return True
    do_q = do_quit
    do_EOF = do_quit


# ---------------------------------------------------------------------------
# __main__ — compile + run + debug an LP file
# ---------------------------------------------------------------------------

def _compile_and_run(prog: LPProgram, entry_args: list[int]):
    """Compile `prog` with tracing enabled, run the `run` export, and
    return (result, trace_cells). Requires wasmtime at runtime."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wasm'))
    import wasmtime
    from wasmtime import _ffi as _wt_ffi

    from lp_pipeline import lp_compile
    wasm_bytes = lp_compile(prog, trace=True)

    cfg = wasmtime.Config()
    _wt_ffi.wasmtime_config_wasm_gc_set(cfg.ptr(), True)
    _wt_ffi.wasmtime_config_wasm_function_references_set(cfg.ptr(), True)
    cfg.wasm_reference_types = True
    cfg.wasm_tail_call = True
    cfg.wasm_multi_value = True

    engine = wasmtime.Engine(cfg)
    store = wasmtime.Store(engine)
    module = wasmtime.Module(engine, wasm_bytes)
    instance = wasmtime.Instance(store, module, [])
    exports = {e.name: instance.exports(store)[e.name]
               for e in module.exports}

    def call(name, *args):
        return exports[name](store, *args)

    result = call("run", *entry_args)
    n = call("__trace_len")
    cells = [call("__trace_get", i) for i in range(n)]
    return result, cells


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Interactive LP Form trace debugger.")
    p.add_argument("lp_file", help="path to an LP source file")
    p.add_argument("--entry",
                   help="name of the entry procedure (default: first proc)")
    p.add_argument("--args", nargs="*", type=int, default=[],
                   help="integer arguments to pass to the entry procedure")
    p.add_argument("--snapshot-interval", type=int, default=500,
                   help="take a state snapshot every N records (0 disables)")
    args = p.parse_args(argv)

    from lp_parser import parse_lp
    with open(args.lp_file) as f:
        source = f.read()
    prog = parse_lp(source, entry=args.entry)

    result, cells = _compile_and_run(prog, args.args)
    trace = Trace(cells, prog)

    print(f"run returned: {result}")
    print(f"trace: {len(trace)} records, {len(cells)} cells")

    replay = Replay(prog, trace, snapshot_interval=args.snapshot_interval)
    DebuggerCLI(replay).cmdloop()


if __name__ == "__main__":
    main()
