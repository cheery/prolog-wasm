#!/usr/bin/env python3
"""
Warren Abstract Machine (WAM) implementation in Python.

Based on "Warren's Abstract Machine: A Tutorial Reconstruction"
by Hassan Aït-Kaci.

Memory layout:
  HEAP:  term storage (grows upward)
  STACK: environments + choice points (grows upward)
  TRAIL: addresses of variables bound since last choice point
  PDL:   push-down list used by unification

Registers:
  H:   heap pointer (next free heap cell)
  S:   structure/subterm pointer
  E:   current environment (stack address)
  P:   program counter (index within current code block)
  CP:  continuation point (return address: label, offset)
  B:   latest choice point (stack address, 0 = none)
  B0:  cut pointer
  HB:  heap backtracking point (H at time of last choice point)
  TR:  trail pointer
  X1..Xn: argument/temporary registers

Cell types:
  REF(addr)  - reference cell; if addr == self, unbound variable
  STR(addr)  - structure cell; addr points to functor cell on heap
  CON(value) - constant (atom or number)
  LIS(addr)  - list cell; addr points to car/cdr pair on heap
  FUN(n/a)   - functor cell (name, arity); not used as standalone tag
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, Any


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

REF = "REF"
STR = "STR"
CON = "CON"
LIS = "LIS"
FUN = "FUN"


@dataclass
class Cell:
    tag: str
    value: Any

    def __repr__(self):
        if self.tag == FUN:
            return f"FUN({self.value[0]}/{self.value[1]})"
        return f"{self.tag}({self.value})"


@dataclass
class Instruction:
    opcode: str
    args: list = field(default_factory=list)

    def __repr__(self):
        if self.args:
            return f"{self.opcode}({', '.join(repr(a) for a in self.args)})"
        return self.opcode


# ---------------------------------------------------------------------------
# Choice Point record
# ---------------------------------------------------------------------------

@dataclass
class ChoicePoint:
    """A choice point saved on backtracking."""
    saved_args: dict        # X registers
    saved_E: int           # environment
    saved_CP: tuple        # (label, offset) continuation
    saved_B: int           # previous choice point
    next_alt: int          # index of next alternative clause
    pred_key: str          # which predicate
    saved_TR: int          # trail pointer
    saved_H: int           # heap pointer
    saved_B0: int          # cut pointer
    saved_call_stack: list # call stack snapshot


# ---------------------------------------------------------------------------
# WAM Machine
# ---------------------------------------------------------------------------

class WAM:
    def __init__(self):
        # Memory
        self.HEAP: list[Cell] = []
        self.STACK: list[Cell] = []  # environments
        self.TRAIL: list[int] = []
        self.PDL: list[int] = []

        # Registers
        self.H: int = 0
        self.S: int = 0
        self.E: int = 0
        self.B: int = 0
        self.B0: int = 0
        self.HB: int = 0
        self.TR: int = 0
        self.num_of_args: int = 0
        self.mode: str = "write"

        # Argument / temporary registers (1-indexed)
        self.X: dict[int, int] = {}

        # Code: label -> list of instructions
        self.code_labels: dict[str, list[Instruction]] = {}

        # Execution state
        self.fail: bool = False
        self.halted: bool = False

        # Current code being executed
        self.current_label: str = ""
        self.current_code: list[Instruction] = []
        self.P: int = 0

        # Continuation: (label, offset)
        self.CP: tuple = ("", 0)

        # Call stack for nested calls
        self.call_stack: list[tuple[str, list[Instruction]]] = []

        # Choice point stack (separate from STACK for clarity)
        self.choice_points: list[ChoicePoint] = []
        # Maps B value to choice_points index; B=0 means none
        self._cp_index: int = 0

        # Query variable tracking
        self.query_vars: dict[str, int] = {}

        # Clause indexing: pred_key -> list of clause labels
        self.pred_clauses: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Heap operations
    # ------------------------------------------------------------------

    def heap_get(self, addr: int) -> Cell:
        return self.HEAP[addr]

    def heap_set(self, addr: int, cell: Cell):
        self.HEAP[addr] = cell

    def heap_push(self, cell: Cell) -> int:
        addr = self.H
        if addr < len(self.HEAP):
            self.HEAP[addr] = cell
        else:
            self.HEAP.append(cell)
        self.H += 1
        return addr

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def stack_push(self, cell: Cell) -> int:
        addr = len(self.STACK)
        self.STACK.append(cell)
        return addr

    def stack_get(self, addr: int) -> Cell:
        return self.STACK[addr]

    def stack_set(self, addr: int, cell: Cell):
        self.STACK[addr] = cell

    # ------------------------------------------------------------------
    # Register access
    # ------------------------------------------------------------------

    def get_reg(self, n: int) -> int:
        return self.X.get(n, 0)

    def set_reg(self, n: int, addr: int):
        self.X[n] = addr

    def get_var(self, v: str) -> int:
        """Get value of variable register Xi or Yi."""
        if v.startswith("X"):
            return self.get_reg(int(v[1:]))
        elif v.startswith("A"):
            return self.get_reg(int(v[1:]))
        elif v.startswith("Y"):
            n = int(v[1:])
            # Env layout: E+0=CE, E+1=CP_label_index, E+2=num_Y, E+3..Yn
            return self.stack_get(self.E + 2 + n).value
        raise RuntimeError(f"Unknown register: {v}")

    def set_var(self, v: str, addr: int):
        if v.startswith("X"):
            self.set_reg(int(v[1:]), addr)
        elif v.startswith("A"):
            self.set_reg(int(v[1:]), addr)
        elif v.startswith("Y"):
            n = int(v[1:])
            self.stack_set(self.E + 2 + n, Cell(REF, addr))
        else:
            raise RuntimeError(f"Unknown register: {v}")

    # ------------------------------------------------------------------
    # Ancillary operations (Appendix B)
    # ------------------------------------------------------------------

    def deref(self, addr: int) -> int:
        while True:
            cell = self.heap_get(addr)
            if cell.tag == REF and cell.value != addr:
                addr = cell.value
            else:
                return addr

    def bind(self, a1: int, a2: int):
        c1 = self.heap_get(a1)
        c2 = self.heap_get(a2)
        if c1.tag == REF and (c2.tag != REF or a2 < a1):
            self.heap_set(a1, Cell(c2.tag, c2.value))
            self._trail(a1)
        else:
            self.heap_set(a2, Cell(c1.tag, c1.value))
            self._trail(a2)

    def _trail(self, addr: int):
        if addr < self.HB:
            self.TRAIL.append(addr)
            self.TR += 1

    def _unwind_trail(self, tr_ptr: int):
        while self.TR > tr_ptr:
            self.TR -= 1
            addr = self.TRAIL[self.TR]
            if addr < len(self.HEAP):
                self.heap_set(addr, Cell(REF, addr))
        del self.TRAIL[tr_ptr:]

    def unify(self, a1: int, a2: int):
        self.PDL.clear()
        self.PDL.append(a1)
        self.PDL.append(a2)
        self.fail = False

        while self.PDL and not self.fail:
            d1 = self.deref(self.PDL.pop())
            d2 = self.deref(self.PDL.pop())
            if d1 != d2:
                c1 = self.heap_get(d1)
                c2 = self.heap_get(d2)
                if c1.tag == REF:
                    self.bind(d1, d2)
                elif c2.tag == REF:
                    self.bind(d2, d1)
                elif c1.tag == CON and c2.tag == CON:
                    if c1.value != c2.value:
                        self.fail = True
                elif c1.tag == LIS and c2.tag == LIS:
                    self.PDL.append(c1.value)
                    self.PDL.append(c2.value)
                    self.PDL.append(c1.value + 1)
                    self.PDL.append(c2.value + 1)
                elif c1.tag == STR and c2.tag == STR:
                    fc1 = self.heap_get(c1.value)
                    fc2 = self.heap_get(c2.value)
                    if fc1.value != fc2.value:
                        self.fail = True
                    else:
                        _, arity = fc1.value
                        for i in range(1, arity + 1):
                            self.PDL.append(c1.value + i)
                            self.PDL.append(c2.value + i)
                else:
                    self.fail = True

    # ------------------------------------------------------------------
    # Instruction execution
    # ------------------------------------------------------------------

    def fetch_and_execute(self):
        if self.halted or self.fail:
            return
        if self.P >= len(self.current_code):
            self.halted = True
            return
        instr = self.current_code[self.P]
        self.P += 1
        self._execute(instr)

    def _execute(self, instr: Instruction):
        op = instr.opcode
        a = instr.args

        # ---- put instructions (query head, body goal arguments) ----

        if op == "put_variable":
            vn, ai = a
            addr = self.heap_push(Cell(REF, self.H))
            self.heap_set(addr, Cell(REF, addr))  # self-referential
            if vn.startswith("Y"):
                self.set_var(vn, addr)
            else:
                self.set_reg(int(vn[1:]), addr)
            self.set_reg(ai, addr)

        elif op == "put_value":
            vn, ai = a
            self.set_reg(ai, self.get_var(vn))

        elif op == "put_unsafe_value":
            yn, ai = a
            addr = self.deref(self.get_var(yn))
            cell = self.heap_get(addr)
            if cell.tag == REF and addr < self.E:
                new_addr = self.heap_push(Cell(REF, self.H))
                self.heap_set(new_addr, Cell(REF, new_addr))
                self.heap_set(addr, Cell(REF, new_addr))
                self._trail(addr)
                self.set_reg(ai, new_addr)
            else:
                self.set_reg(ai, addr)

        elif op == "put_structure":
            (name, arity), ai = a
            fun_addr = self.H + 1          # FUN will land at H+1 after STR is pushed
            str_addr = self.heap_push(Cell(STR, fun_addr))   # STR at H
            self.heap_push(Cell(FUN, (name, arity)))          # FUN at H+1
            self.set_reg(ai, str_addr)
            self.mode = "write"

        elif op == "put_list":
            ai = a[0]
            car_addr = self.H + 1
            lis_addr = self.heap_push(Cell(LIS, car_addr))
            self.S = self.H
            self.set_reg(ai, lis_addr)
            self.mode = "write"

        elif op == "put_constant":
            c, ai = a
            addr = self.heap_push(Cell(CON, c))
            self.set_reg(ai, addr)

        # ---- get instructions (program clause head matching) ----

        elif op == "get_variable":
            vn, ai = a
            if vn.startswith("Y"):
                self.set_var(vn, self.get_reg(ai))
            else:
                self.set_reg(int(vn[1:]), self.get_reg(ai))

        elif op == "get_value":
            vn, ai = a
            self.unify(self.get_var(vn), self.get_reg(ai))

        elif op == "get_structure":
            (name, arity), ai = a
            addr = self.deref(self.get_reg(ai))
            cell = self.heap_get(addr)
            if cell.tag == REF:
                fun_addr = self.H + 1          # FUN will land at H+1 after STR is pushed
                str_addr = self.heap_push(Cell(STR, fun_addr))   # STR at H
                self.heap_push(Cell(FUN, (name, arity)))          # FUN at H+1
                self.heap_set(addr, Cell(REF, str_addr))
                self._trail(addr)
                self.mode = "write"
            elif cell.tag == STR:
                fun_cell = self.heap_get(cell.value)
                if fun_cell.value == (name, arity):
                    self.S = cell.value + 1    # cell.value = fun_addr; args start at fun_addr+1
                    self.mode = "read"
                else:
                    self.fail = True
            else:
                self.fail = True

        elif op == "get_list":
            ai = a[0]
            addr = self.deref(self.get_reg(ai))
            cell = self.heap_get(addr)
            if cell.tag == REF:
                car_addr = self.H + 1
                lis_addr = self.heap_push(Cell(LIS, car_addr))
                self.heap_set(addr, Cell(REF, lis_addr))
                self._trail(addr)
                self.S = self.H
                self.mode = "write"
            elif cell.tag == LIS:
                self.S = cell.value
                self.mode = "read"
            else:
                self.fail = True

        elif op == "get_constant":
            c, ai = a
            addr = self.deref(self.get_reg(ai))
            cell = self.heap_get(addr)
            if cell.tag == REF:
                c_addr = self.heap_push(Cell(CON, c))
                self.heap_set(addr, Cell(REF, c_addr))
                self._trail(addr)
            elif cell.tag == CON and cell.value == c:
                pass
            else:
                self.fail = True

        # ---- set instructions (structure building, write mode) ----

        elif op == "set_variable":
            vn = a[0]
            addr = self.heap_push(Cell(REF, self.H))
            self.heap_set(addr, Cell(REF, addr))
            if vn.startswith("Y"):
                self.set_var(vn, addr)
            else:
                self.set_reg(int(vn[1:]), addr)

        elif op == "set_value":
            vn = a[0]
            src_addr = self.get_var(vn)
            src = self.heap_get(src_addr)
            self.heap_push(Cell(src.tag, src.value))

        elif op == "set_local_value":
            vn = a[0]
            addr = self.deref(self.get_var(vn))
            cell = self.heap_get(addr)
            if cell.tag == REF and addr < self.E:
                new_addr = self.heap_push(Cell(REF, self.H))
                self.heap_set(new_addr, Cell(REF, new_addr))
                self.heap_set(addr, Cell(REF, new_addr))
                self._trail(addr)
            else:
                self.heap_push(Cell(cell.tag, cell.value))

        elif op == "set_constant":
            self.heap_push(Cell(CON, a[0]))

        elif op == "set_void":
            for _ in range(a[0]):
                addr = self.heap_push(Cell(REF, self.H))
                self.heap_set(addr, Cell(REF, addr))

        # ---- unify instructions (structure matching, read/write) ----

        elif op == "unify_variable":
            vn = a[0]
            if self.mode == "read":
                addr = self.S
                if vn.startswith("Y"):
                    self.set_var(vn, addr)
                else:
                    self.set_reg(int(vn[1:]), addr)
            else:
                addr = self.heap_push(Cell(REF, self.H))
                self.heap_set(addr, Cell(REF, addr))
                if vn.startswith("Y"):
                    self.set_var(vn, addr)
                else:
                    self.set_reg(int(vn[1:]), addr)
            self.S += 1

        elif op == "unify_value":
            vn = a[0]
            if self.mode == "read":
                self.unify(self.get_var(vn), self.S)
            else:
                src_addr = self.get_var(vn)
                src = self.heap_get(src_addr)
                self.heap_push(Cell(src.tag, src.value))
            self.S += 1

        elif op == "unify_local_value":
            vn = a[0]
            if self.mode == "read":
                self.unify(self.get_var(vn), self.S)
            else:
                addr = self.deref(self.get_var(vn))
                cell = self.heap_get(addr)
                if cell.tag == REF and addr < self.E:
                    new_addr = self.heap_push(Cell(REF, self.H))
                    self.heap_set(new_addr, Cell(REF, new_addr))
                    self.heap_set(addr, Cell(REF, new_addr))
                    self._trail(addr)
                else:
                    self.heap_push(Cell(cell.tag, cell.value))
            self.S += 1

        elif op == "unify_constant":
            c = a[0]
            if self.mode == "read":
                addr = self.deref(self.S)
                cell = self.heap_get(addr)
                if cell.tag == REF:
                    c_addr = self.heap_push(Cell(CON, c))
                    self.heap_set(addr, Cell(REF, c_addr))
                    self._trail(addr)
                elif cell.tag == CON and cell.value == c:
                    pass
                else:
                    self.fail = True
            else:
                self.heap_push(Cell(CON, c))
            self.S += 1

        elif op == "unify_void":
            for _ in range(a[0]):
                if self.mode == "write":
                    addr = self.heap_push(Cell(REF, self.H))
                    self.heap_set(addr, Cell(REF, addr))
                self.S += 1

        # ---- control instructions ----

        elif op == "allocate":
            n = a[0]
            new_e = len(self.STACK)
            self.stack_push(Cell(REF, self.E))      # CE
            self.stack_push(Cell(REF, 0))            # CP (placeholder)
            self.stack_push(Cell(CON, n))            # num permanent vars
            for i in range(n):
                addr = new_e + 3 + i
                self.stack_push(Cell(REF, addr))     # Yi = unbound
            self.E = new_e

        elif op == "deallocate":
            ce = self.stack_get(self.E).value
            n = int(self.stack_get(self.E + 2).value)
            # Don't physically delete; just move E back.
            # The choice point mechanism handles cleanup on backtrack.
            self.E = ce

        elif op == "call":
            (name, arity) = a[0]
            key = f"{name}/{arity}"
            self.num_of_args = arity
            # Save continuation
            self.call_stack.append((self.current_label, list(self.current_code)))
            self.CP = (self.current_label, self.P)
            self.P = 0
            if key not in self.code_labels:
                self.fail = True
                return
            self.current_label = key
            self.current_code = self.code_labels[key]
            self.HB = self.H

        elif op == "execute":
            (name, arity) = a[0]
            key = f"{name}/{arity}"
            self.num_of_args = arity
            # Tail call: don't save continuation (reuse current CP)
            self.P = 0
            if key not in self.code_labels:
                self.fail = True
                return
            self.current_label = key
            self.current_code = self.code_labels[key]
            self.HB = self.H

        elif op == "proceed":
            label, offset = self.CP
            if not label and offset == 0:
                self.halted = True
                return
            if self.call_stack:
                _, saved_code = self.call_stack.pop()
                self.current_label = label
                self.current_code = saved_code
            self.P = offset
            self.CP = ("", 0)

        # ---- choice instructions ----

        elif op == "try_me_else":
            # Create a choice point; on backtracking, try next clause
            label = a[0]  # label of next alternative
            self._push_choice(label)

        elif op == "retry_me_else":
            label = a[0]
            self._retry_choice(label)

        elif op == "trust_me":
            self._trust_choice()

        elif op == "try":
            label = a[0]
            self._push_choice(None)
            self._jump_to_label(label)

        elif op == "retry":
            label = a[0]
            self._retry_choice(None)
            self._jump_to_label(label)

        elif op == "trust":
            label = a[0]
            self._trust_choice()
            self._jump_to_label(label)

        # ---- indexing ----

        elif op == "switch_on_term":
            var_label, con_label, lis_label, str_label = a
            addr = self.deref(self.get_reg(1))
            cell = self.heap_get(addr)
            if cell.tag == REF:
                target = var_label
            elif cell.tag == CON:
                target = con_label
            elif cell.tag == LIS:
                target = lis_label
            elif cell.tag == STR:
                target = str_label
            else:
                self.fail = True
                return
            if target is not None:
                self._jump_to_label(target)
            else:
                self.fail = True

        elif op == "switch_on_constant":
            cases = a[0]
            addr = self.deref(self.get_reg(1))
            cell = self.heap_get(addr)
            if cell.tag == CON:
                for c, label in cases:
                    if cell.value == c:
                        self._jump_to_label(label)
                        return
            self.fail = True

        elif op == "switch_on_structure":
            cases = a[0]
            addr = self.deref(self.get_reg(1))
            cell = self.heap_get(addr)
            if cell.tag == STR:
                fun_cell = self.heap_get(cell.value)
                for functor, label in cases:
                    if fun_cell.value == functor:
                        self._jump_to_label(label)
                        return
            self.fail = True

        # ---- cut ----

        elif op == "neck_cut":
            self._cut_to(self.B0)

        elif op == "get_level":
            yn = a[0]
            n = int(yn[1:])
            self.stack_set(self.E + 2 + n, Cell(CON, self.B0))

        elif op == "cut":
            yn = a[0]
            n = int(yn[1:])
            cutoff = int(self.stack_get(self.E + 2 + n).value)
            self._cut_to(cutoff)

        else:
            raise RuntimeError(f"Unknown instruction: {op}")

    # ------------------------------------------------------------------
    # Choice point operations
    # ------------------------------------------------------------------

    def _push_choice(self, next_label: Optional[str]):
        cp = ChoicePoint(
            saved_args=dict(self.X),
            saved_E=self.E,
            saved_CP=self.CP,
            saved_B=self.B,
            next_alt=0,
            pred_key=next_label or "",
            saved_TR=self.TR,
            saved_H=self.H,
            saved_B0=self.B0,
            saved_call_stack=list(self.call_stack),
        )
        self.B = len(self.choice_points) + 1
        self.choice_points.append(cp)
        self.B0 = self.B
        self.HB = self.H

    def _retry_choice(self, next_label: Optional[str]):
        if self.B == 0:
            self.fail = True
            return
        cp = self.choice_points[self.B - 1]
        # Restore state
        self.X = dict(cp.saved_args)
        self.E = cp.saved_E
        self.CP = cp.saved_CP
        self._unwind_trail(cp.saved_TR)
        self.H = cp.saved_H
        self.HB = self.H
        self.B0 = cp.saved_B0
        self.call_stack = list(cp.saved_call_stack)
        # Update next alternative
        if next_label:
            cp.pred_key = next_label

    def _trust_choice(self):
        if self.B == 0:
            self.fail = True
            return
        cp = self.choice_points[self.B - 1]
        self.X = dict(cp.saved_args)
        self.E = cp.saved_E
        self.CP = cp.saved_CP
        self._unwind_trail(cp.saved_TR)
        self.H = cp.saved_H
        self.HB = self.H
        self.B0 = cp.saved_B0
        self.call_stack = list(cp.saved_call_stack)
        self.B = cp.saved_B
        # Remove this choice point
        self.choice_points.pop()

    def _cut_to(self, cutoff: int):
        while self.B > cutoff and self.B > 0:
            cp = self.choice_points[self.B - 1]
            self.B = cp.saved_B
            # Keep the choice point around but it won't be reached
        if self.B < cutoff:
            self.B0 = cutoff

    def _jump_to_label(self, label: str):
        if label in self.code_labels:
            self.current_label = label
            self.current_code = self.code_labels[label]
            self.P = 0
        else:
            self.fail = True

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def run(self):
        while not self.halted and not self.fail:
            self.fetch_and_execute()

    def run_query(self) -> list[dict]:
        """Run current query, return all solutions via backtracking."""
        solutions = []
        self.run()

        if not self.fail and self.halted:
            solutions.append(self._collect_solution())

        while self.B > 0 and not self.fail:
            self._backtrack()
            if not self.fail:
                self.run()
                if not self.fail and self.halted:
                    solutions.append(self._collect_solution())

        return solutions

    def _backtrack(self):
        if self.B == 0:
            self.fail = True
            return

        cp = self.choice_points[self.B - 1]
        self.X = dict(cp.saved_args)
        self.E = cp.saved_E
        self.CP = cp.saved_CP
        self._unwind_trail(cp.saved_TR)
        self.H = cp.saved_H
        self.HB = self.H
        self.B0 = cp.saved_B0
        self.call_stack = list(cp.saved_call_stack)
        self.halted = False
        self.fail = False

        # Jump to next alternative clause
        next_label = cp.pred_key
        if next_label and next_label in self.code_labels:
            self.current_label = next_label
            self.current_code = self.code_labels[next_label]
            self.P = 0
            # Advance to next alternative for future backtracking
            cp.next_alt += 1
            clauses = self.pred_clauses.get(self.current_label, [])
            if cp.next_alt < len(clauses):
                cp.pred_key = clauses[cp.next_alt]
            else:
                cp.pred_key = None
        else:
            # No more alternatives; pop this choice point
            prev_B = cp.saved_B
            self.choice_points.pop()
            self.B = prev_B

    # ------------------------------------------------------------------
    # Solution display
    # ------------------------------------------------------------------

    def _collect_solution(self) -> dict:
        result = {}
        for name, addr in self.query_vars.items():
            result[name] = self._display(addr)
        return result

    def _display(self, addr: int) -> Any:
        addr = self.deref(addr)
        cell = self.heap_get(addr)
        if cell.tag == REF:
            return f"_{addr}"
        elif cell.tag == CON:
            return cell.value
        elif cell.tag == LIS:
            car = self._display(cell.value)
            cdr = self._display(cell.value + 1)
            cdr_cell = self.heap_get(self.deref(cell.value + 1))
            if cdr_cell.tag == CON and cdr_cell.value == "[]":
                return [car]
            elif isinstance(cdr, list):
                return [car] + cdr
            else:
                return [car, "|", cdr]
        elif cell.tag == STR:
            fun_cell = self.heap_get(cell.value)
            name, arity = fun_cell.value
            if arity == 0:
                return name
            args = [self._display(cell.value + i) for i in range(1, arity + 1)]
            return f"{name}({', '.join(str(a) for a in args)})"
        return f"?{addr}"

    # ------------------------------------------------------------------
    # Reset (keep compiled code)
    # ------------------------------------------------------------------

    def reset(self):
        self.HEAP.clear()
        self.STACK.clear()
        self.TRAIL.clear()
        self.PDL.clear()
        self.H = 0
        self.S = 0
        self.E = 0
        self.P = 0
        self.CP = ("", 0)
        self.B = 0
        self.B0 = 0
        self.HB = 0
        self.TR = 0
        self.num_of_args = 0
        self.mode = "write"
        self.X.clear()
        self.fail = False
        self.halted = False
        self.current_label = ""
        self.current_code = []
        self.call_stack.clear()
        self.choice_points.clear()
        self.query_vars.clear()


# ==========================================================================
# PROLOG PARSER
# ==========================================================================

@dataclass
class Term:
    type: str  # "atom", "num", "var", "compound"
    value: Any = None
    args: list = field(default_factory=list)

    @staticmethod
    def atom(name): return Term("atom", name)
    @staticmethod
    def num(val): return Term("num", val)
    @staticmethod
    def var(name): return Term("var", name)
    @staticmethod
    def compound(functor, args): return Term("compound", functor, args)

    def is_var(self): return self.type == "var"
    def is_atom(self): return self.type == "atom"
    def is_compound(self): return self.type == "compound"
    def is_num(self): return self.type == "num"

    def functor(self):
        return self.value if self.is_compound() else self.value

    def arity(self):
        return len(self.args) if self.is_compound() else 0

    def name_arity(self):
        return (self.value, len(self.args) if self.is_compound() else 0)

    def __repr__(self):
        if self.is_var(): return self.value
        if self.is_atom(): return str(self.value)
        if self.is_num(): return str(self.value)
        if self.is_compound():
            if self.value == "." and len(self.args) == 2:
                return self._repr_list()
            return f"{self.value}({', '.join(repr(a) for a in self.args)})"
        return f"?{self.type}"

    def _repr_list(self):
        elems = []
        cur = self
        while cur.is_compound() and cur.value == "." and len(cur.args) == 2:
            elems.append(repr(cur.args[0]))
            cur = cur.args[1]
        if cur.is_atom() and cur.value == "[]":
            return f"[{', '.join(elems)}]"
        return f"[{', '.join(elems)}|{repr(cur)}]"


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


def tokenize(text: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]

        if c in " \t\n\r":
            i += 1
            continue

        # Comments
        if c == "%":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if i + 1 < len(text) and c == "/" and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        # Special punctuation
        if text[i:i+2] == ":-":
            tokens.append(Token("PUNCT", ":-"))
            i += 2
            continue
        if text[i:i+2] == "->":
            tokens.append(Token("PUNCT", "->"))
            i += 2
            continue
        if text[i:i+2] == "\\+":
            tokens.append(Token("ATOM", "\\+"))
            i += 2
            continue
        if text[i:i+3] == "=..":
            tokens.append(Token("ATOM", "=.."))
            i += 3
            continue

        if c in "(),.[]{};|!":
            tokens.append(Token("PUNCT", c))
            i += 1
            continue

        # Quoted atom
        if c == "'":
            j = i + 1
            while j < len(text) and text[j] != "'":
                if text[j] == "\\": j += 1
                j += 1
            tokens.append(Token("ATOM", text[i+1:j]))
            i = j + 1
            continue

        # Number
        if c.isdigit() or (c == "-" and i + 1 < len(text) and text[i+1].isdigit()):
            j = i + 1 if c == "-" else i
            while j < len(text) and (text[j].isdigit() or text[j] == "."):
                j += 1
            val = text[i:j]
            tokens.append(Token("NUM", val))
            i = j
            continue

        # Variable
        if c.isupper() or c == "_":
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("VAR", text[i:j]))
            i = j
            continue

        # Atom
        if c.islower() or c in "+-*/\\<>=~&^$#@":
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("ATOM", text[i:j]))
            i = j
            continue

        raise SyntaxError(f"Unexpected char: {c!r} at pos {i}")
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token("EOF", None)

    def advance(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, value=None):
        t = self.advance()
        if value and t.value != value:
            raise SyntaxError(f"Expected {value!r}, got {t.value!r}")
        return t

    def parse_term(self):
        return self.parse_comma()

    def parse_arg(self):
        """Parse a single argument (no comma handling — commas are separators in arg lists)."""
        return self.parse_primary()

    def parse_comma(self):
        left = self.parse_primary()
        while self.peek().value == ",":
            self.advance()
            right = self.parse_primary()
            left = Term.compound(",", [left, right])
        return left

    def parse_primary(self):
        tok = self.peek()

        if tok.type == "NUM":
            self.advance()
            try:
                return Term.num(int(tok.value))
            except ValueError:
                return Term.num(float(tok.value))

        if tok.type == "VAR":
            self.advance()
            return Term.var(tok.value)

        if tok.type == "ATOM":
            self.advance()
            name = tok.value
            if self.peek().value == "(":
                self.advance()
                args = []
                if self.peek().value != ")":
                    args.append(self.parse_arg())
                    while self.peek().value == ",":
                        self.advance()
                        args.append(self.parse_arg())
                self.expect(")")
                return Term.compound(name, args)
            return Term.atom(name)

        if tok.type == "PUNCT" and tok.value == "[":
            return self.parse_list()

        if tok.type == "PUNCT" and tok.value == "(":
            self.advance()
            t = self.parse_term()
            self.expect(")")
            return t

        if tok.type == "PUNCT" and tok.value == "!":
            self.advance()
            return Term.atom("!")

        raise SyntaxError(f"Unexpected: {tok}")

    def parse_list(self):
        self.expect("[")
        elements = []
        if self.peek().value != "]":
            elements.append(self.parse_arg())
            while self.peek().value == ",":
                self.advance()
                elements.append(self.parse_arg())
        tail = None
        if self.peek().value == "|":
            self.advance()
            tail = self.parse_arg()
        self.expect("]")
        # Build list as ./2 terms
        t = tail if tail else Term.atom("[]")
        for e in reversed(elements):
            t = Term.compound(".", [e, t])
        return t


def parse_clause(text: str):
    """Parse a clause. Returns (head, body_or_None)."""
    tokens = tokenize(text.strip().rstrip("."))
    if not tokens:
        return None
    parser = Parser(tokens)
    head = parser.parse_term()
    if parser.peek().value == ":-":
        parser.advance()
        body = parser.parse_term()
        return (head, body)
    return (head, None)


# ==========================================================================
# PROLOG -> WAM COMPILER
# ==========================================================================

class Compiler:
    def __init__(self, wam: WAM):
        self.wam = wam

    # ------------------------------------------------------------------
    # Variable analysis
    # ------------------------------------------------------------------

    def _collect_vars(self, term, vs: set):
        if term is None: return
        if term.is_var():
            vs.add(term.value)
        elif term.is_compound():
            for a in term.args:
                self._collect_vars(a, vs)

    def _vars_in(self, term) -> set:
        vs = set()
        self._collect_vars(term, vs)
        return vs

    def _all_clause_vars(self, head, body) -> set:
        vs = set()
        self._collect_vars(head, vs)
        self._collect_vars(body, vs)
        return vs

    def _flatten_conj(self, body) -> list:
        if body.is_compound() and body.value == ",":
            return self._flatten_conj(body.args[0]) + self._flatten_conj(body.args[1])
        return [body]

    def _var_occurrences(self, body) -> dict:
        """Count how many goals each variable appears in."""
        counts = {}
        if body is None:
            return counts
        for g in self._flatten_conj(body):
            for v in self._vars_in(g):
                counts[v] = counts.get(v, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Register allocation
    # ------------------------------------------------------------------

    def _allocate_registers(self, head, body):
        """Allocate Xi (temp) and Yi (permanent) registers for a clause.
        Returns (reg_map, num_permanent)."""
        all_vars = self._all_clause_vars(head, body)
        if not all_vars:
            return {}, 0

        # Permanent vars: any variable that appears in a body goal AFTER the
        # first one. Such variables must survive the first call's register
        # clobbering of A1..An.
        if body:
            goals = self._flatten_conj(body)
            later_vars = set()
            for g in goals[1:]:
                later_vars.update(self._vars_in(g))
            perm = later_vars
        else:
            perm = set()

        perm_list = sorted(perm)
        num_perm = len(perm_list)
        perm_map = {}
        for i, v in enumerate(perm_list):
            perm_map[v] = f"Y{i + 1}"

        arity = len(head.args) if head.is_compound() else 0

        # Register allocation:
        # - If head arg i is a bare variable V, V gets register Xi (= Ai)
        # - Variables inside structures/lists get temp registers above arity
        reg_map = {}
        counter = arity  # next available temp register

        for i, arg in enumerate(head.args if head.is_compound() else []):
            if arg.is_var():
                # Bare variable argument: gets the argument register
                if arg.value not in reg_map:
                    reg_map[arg.value] = f"X{i + 1}"
            else:
                # Compound/constant argument: sub-variables get temps
                for v in sorted(self._vars_in(arg)):
                    if v not in reg_map:
                        counter += 1
                        reg_map[v] = f"X{counter}"

        # Remaining vars (only in body, not in head)
        for v in sorted(all_vars):
            if v not in reg_map:
                counter += 1
                reg_map[v] = f"X{counter}"

        # Override with permanent registers
        for v in perm:
            if v in reg_map:
                reg_map[v] = perm_map[v]

        return reg_map, num_perm

    # ------------------------------------------------------------------
    # Clause compilation
    # ------------------------------------------------------------------

    def compile_clause(self, head, body, reg_map, num_perm) -> list[Instruction]:
        instrs = []

        # Allocate environment BEFORE head matching (so Y registers exist)
        if body and num_perm > 0:
            instrs.append(Instruction("allocate", [num_perm]))

        # Head matching: get_* instructions
        # Track first-occurrence variables across all head arguments
        head_seen = set()
        if head.is_compound():
            for i, arg in enumerate(head.args):
                self._emit_get_arg(arg, i + 1, reg_map, instrs, head_seen)

        # Body
        if body:
            # Track which variables have been initialized (assigned a value)
            # All head variables are initialized by get_* instructions
            seen_vars = set(head_seen)

            goals = self._flatten_conj(body)
            for gi, goal in enumerate(goals):
                is_last = (gi == len(goals) - 1)
                self._emit_goal(goal, reg_map, instrs, is_last, seen_vars)

        if not body or num_perm == 0:
            instrs.append(Instruction("proceed", []))
        # For body clauses with env: deallocate was already emitted before execute
        # But if there's no body goal using execute (shouldn't happen for rules),
        # add proceed anyway as safety
        if body and num_perm == 0:
            pass  # proceed already added above

        return instrs

    def _emit_get_arg(self, arg, ai, reg_map, instrs, seen=None):
        """Emit get_* instructions for a head argument."""
        if seen is None:
            seen = set()
        if arg.is_var():
            vn = reg_map.get(arg.value, f"X{ai}")
            if arg.value in seen:
                instrs.append(Instruction("get_value", [vn, ai]))
            else:
                instrs.append(Instruction("get_variable", [vn, ai]))
                seen.add(arg.value)
        elif arg.is_atom() or arg.is_num():
            instrs.append(Instruction("get_constant", [arg.value, ai]))
        elif arg.is_compound() and arg.value == "." and len(arg.args) == 2:
            instrs.append(Instruction("get_list", [ai]))
            self._emit_unify_list(arg, reg_map, instrs, seen)
        elif arg.is_compound():
            name, arity = arg.value, len(arg.args)
            instrs.append(Instruction("get_structure", [(name, arity), ai]))
            for sub in arg.args:
                self._emit_unify_sub(sub, reg_map, instrs, seen)

    def _emit_unify_list(self, term, reg_map, instrs, seen):
        """Emit unify_* instructions for list elements."""
        cur = term
        while cur.is_compound() and cur.value == "." and len(cur.args) == 2:
            self._emit_unify_sub(cur.args[0], reg_map, instrs, seen)
            cur = cur.args[1]
        if not (cur.is_atom() and cur.value == "[]"):
            self._emit_unify_sub(cur, reg_map, instrs, seen)

    def _emit_unify_sub(self, sub, reg_map, instrs, first_occurrence: set):
        """Emit a single unify_* instruction."""
        if sub.is_var():
            vn = reg_map.get(sub.value)
            if vn is None:
                instrs.append(Instruction("unify_void", [1]))
            elif sub.value not in first_occurrence:
                instrs.append(Instruction("unify_variable", [vn]))
                first_occurrence.add(sub.value)
            else:
                instrs.append(Instruction("unify_value", [vn]))
        elif sub.is_atom() or sub.is_num():
            instrs.append(Instruction("unify_constant", [sub.value]))
        elif sub.is_compound() and sub.value == "." and len(sub.args) == 2:
            # Nested list - simplified: treat as variable (needs recursive unif)
            # A full compiler would handle this with auxiliary variables
            vn = reg_map.get(repr(sub))
            if vn:
                instrs.append(Instruction("unify_value", [vn]))
            else:
                instrs.append(Instruction("unify_void", [1]))
        elif sub.is_compound():
            instrs.append(Instruction("unify_void", [1]))
        else:
            instrs.append(Instruction("unify_void", [1]))

    def _emit_goal(self, goal, reg_map, instrs, is_last, seen_vars: set):
        """Emit put_* + call/execute for a body goal."""
        if goal.is_atom() and goal.value == "!":
            if is_last:
                instrs.append(Instruction("neck_cut", []))
            else:
                instrs.append(Instruction("get_level", ["Y1"]))
                instrs.append(Instruction("cut", ["Y1"]))
            return

        name = goal.value if goal.is_compound() else goal.value
        arity = len(goal.args) if goal.is_compound() else 0

        if goal.is_compound():
            for i, arg in enumerate(goal.args):
                self._emit_put_arg(arg, i + 1, reg_map, instrs, seen_vars)

        pred = (name, arity)
        if is_last:
            # deallocate before execute (tail call optimization)
            # Check if we have permanent vars by looking for Y registers in reg_map
            has_perm = any(v.startswith("Y") for v in reg_map.values())
            if has_perm:
                instrs.append(Instruction("deallocate", []))
            instrs.append(Instruction("execute", [pred]))
        else:
            instrs.append(Instruction("call", [pred]))

    def _emit_put_arg(self, arg, ai, reg_map, instrs, seen_vars: set):
        """Emit put_* for a body goal argument."""
        if arg.is_var():
            vn = reg_map.get(arg.value, f"X{ai}")
            if arg.value not in seen_vars:
                # First occurrence: create new variable
                seen_vars.add(arg.value)
                instrs.append(Instruction("put_variable", [vn, ai]))
            elif vn.startswith("Y"):
                instrs.append(Instruction("put_unsafe_value", [vn, ai]))
            else:
                instrs.append(Instruction("put_value", [vn, ai]))
        elif arg.is_atom() or arg.is_num():
            instrs.append(Instruction("put_constant", [arg.value, ai]))
        elif arg.is_compound() and arg.value == "." and len(arg.args) == 2:
            instrs.append(Instruction("put_list", [ai]))
            self._emit_set_list(arg, reg_map, instrs, seen_vars)
        elif arg.is_compound():
            name, arity = arg.value, len(arg.args)
            instrs.append(Instruction("put_structure", [(name, arity), ai]))
            for sub in arg.args:
                self._emit_set_sub(sub, reg_map, instrs, seen_vars)

    def _emit_set_list(self, term, reg_map, instrs, seen_vars: set):
        """Emit set_* instructions for building a list in a body goal.

        For multi-element lists, emits intermediate put_list instructions.
        """
        elements = []
        cur = term
        while cur.is_compound() and cur.value == "." and len(cur.args) == 2:
            elements.append(cur.args[0])
            cur = cur.args[1]
        tail = cur

        for i, elem in enumerate(elements):
            if i > 0:
                instrs.append(Instruction("put_list", [0]))
            self._emit_set_sub(elem, reg_map, instrs, seen_vars)

        if tail.is_atom() and tail.value == "[]":
            instrs.append(Instruction("set_constant", ["[]"]))
        elif tail.is_var():
            vn = reg_map.get(tail.value)
            if vn:
                instrs.append(Instruction("set_value", [vn]))
            else:
                instrs.append(Instruction("set_void", [1]))
        else:
            self._emit_set_sub(tail, reg_map, instrs, seen_vars)

    def _emit_set_sub(self, sub, reg_map, instrs, seen_vars: set):
        if sub.is_var():
            vn = reg_map.get(sub.value)
            if vn is None:
                instrs.append(Instruction("set_void", [1]))
            elif sub.value not in seen_vars:
                seen_vars.add(sub.value)
                instrs.append(Instruction("set_variable", [vn]))
            else:
                instrs.append(Instruction("set_value", [vn]))
        elif sub.is_atom() or sub.is_num():
            instrs.append(Instruction("set_constant", [sub.value]))
        elif sub.is_compound() and sub.value == "." and len(sub.args) == 2:
            # Nested list
            self._emit_set_list(sub, reg_map, instrs, seen_vars)
        elif sub.is_compound():
            name, arity = sub.value, len(sub.args)
            instrs.append(Instruction("set_void", [1]))  # simplified
        else:
            instrs.append(Instruction("set_void", [1]))

    # ------------------------------------------------------------------
    # Predicate compilation (multi-clause with choice points)
    # ------------------------------------------------------------------

    def _compile_predicate(self, pred_key, clauses):
        """Compile all clauses for a predicate.

        Correct WAM structure interleaves choice instructions with clause code:
          Clause 1:  try_me_else C2    <head1> <body1> proceed
          Clause 2:  retry_me_else C3  <head2> <body2> proceed  (at label C2)
          Clause 3:  trust_me          <head3> <body3> proceed  (at label C3)
        """
        clause_labels = []

        if len(clauses) == 1:
            # Single clause: no choice point needed
            head, body = clauses[0]
            reg_map, num_perm = self._allocate_registers(head, body)
            instrs = self.compile_clause(head, body, reg_map, num_perm)
            self.wam.code_labels[pred_key] = instrs
            clause_labels.append(pred_key)
        else:
            # Multiple clauses: interleave choice + clause code
            for i, (head, body) in enumerate(clauses):
                label = f"{pred_key}_c{i + 1}"
                reg_map, num_perm = self._allocate_registers(head, body)
                clause_instrs = self.compile_clause(head, body, reg_map, num_perm)

                # Remove the proceed at the end; we'll add our own
                # Actually keep proceed — it's the clause terminator

                full_instrs = []
                # Prepend choice instruction
                if i == 0:
                    full_instrs.append(
                        Instruction("try_me_else", [f"{pred_key}_c{i + 2}"]))
                elif i < len(clauses) - 1:
                    full_instrs.append(
                        Instruction("retry_me_else", [f"{pred_key}_c{i + 2}"]))
                else:
                    full_instrs.append(Instruction("trust_me", []))

                # If clause has allocate, insert it after the choice instruction
                if clause_instrs and clause_instrs[0].opcode == "allocate":
                    full_instrs.append(clause_instrs[0])
                    # Head matching comes next
                    full_instrs.extend(clause_instrs[1:])
                else:
                    full_instrs.extend(clause_instrs)

                self.wam.code_labels[label] = full_instrs
                clause_labels.append(label)

            # Main entry points to first clause label
            first_label = clause_labels[0]
            self.wam.code_labels[pred_key] = list(self.wam.code_labels[first_label])

        self.wam.pred_clauses[pred_key] = clause_labels

    def _label_to_pred(self, label):
        """Convert label back to (name, arity) for execute."""
        # Remove _cN suffix
        parts = label.rsplit("_c", 1)
        pred_key = parts[0]
        if "/" in pred_key:
            name, ar = pred_key.rsplit("/", 1)
            return (name, int(ar))
        return (pred_key, 0)

    # ------------------------------------------------------------------
    # Program loading
    # ------------------------------------------------------------------

    def load_program(self, source: str):
        """Parse and compile a Prolog program."""
        # Tokenize the whole source, then split on '.' (PUNCT) tokens so that
        # multiple clauses on the same line are handled correctly.
        clauses_by_pred = {}
        try:
            all_tokens = tokenize(source)
        except SyntaxError:
            return

        clause_tokens: list = []
        for tok in all_tokens:
            if tok.type == "PUNCT" and tok.value == ".":
                if clause_tokens:
                    parser = Parser(clause_tokens)
                    try:
                        head = parser.parse_term()
                        body = None
                        if parser.peek().value == ":-":
                            parser.advance()
                            body = parser.parse_term()
                        key = f"{head.value}/{len(head.args) if head.is_compound() else 0}"
                        clauses_by_pred.setdefault(key, []).append((head, body))
                    except SyntaxError:
                        pass
                    clause_tokens = []
            else:
                clause_tokens.append(tok)

        for pred_key, clauses in clauses_by_pred.items():
            self._compile_predicate(pred_key, clauses)

    # ------------------------------------------------------------------
    # Query compilation
    # ------------------------------------------------------------------

    def compile_query(self, query_text: str) -> list[Instruction]:
        """Compile a query into WAM instructions."""
        query_text = query_text.strip().rstrip(".")
        if not query_text:
            return [], {}
        tokens = tokenize(query_text)
        parser = Parser(tokens)
        query_term = parser.parse_term()

        goals = self._flatten_conj(query_term)

        # Collect all variables in the query
        all_vars = set()
        for g in goals:
            self._collect_vars(g, all_vars)

        # Assign registers starting ABOVE the max arity to avoid collision
        # with argument registers A1..An used by put_* instructions.
        max_arity = 0
        for g in goals:
            if g.is_compound():
                max_arity = max(max_arity, len(g.args))

        reg_map = {}
        var_list = sorted(all_vars)
        for i, v in enumerate(var_list):
            reg_map[v] = f"X{max_arity + i + 1}"

        instrs = []

        # For each goal, put arguments and call
        seen_vars: set = set()
        for goal in goals:
            if goal.is_compound():
                name = goal.value
                arity = len(goal.args)
                for i, arg in enumerate(goal.args):
                    self._emit_query_put(arg, i + 1, reg_map, instrs, seen_vars)
                instrs.append(Instruction("call", [(name, arity)]))
            elif goal.is_atom():
                instrs.append(Instruction("call", [(goal.value, 0)]))

        return instrs, reg_map

    def _emit_query_put(self, arg, ai, reg_map, instrs, seen_vars: set = None):
        """Emit put_* instructions for query arguments."""
        if seen_vars is None:
            seen_vars = set()
        if arg.is_var():
            vn = reg_map.get(arg.value, f"X{ai}")
            if arg.value not in seen_vars:
                seen_vars.add(arg.value)
                instrs.append(Instruction("put_variable", [vn, ai]))
            else:
                instrs.append(Instruction("put_value", [vn, ai]))
        elif arg.is_atom() or arg.is_num():
            instrs.append(Instruction("put_constant", [arg.value, ai]))
        elif arg.is_compound() and arg.value == "." and len(arg.args) == 2:
            instrs.append(Instruction("put_list", [ai]))
            self._emit_query_set_list(arg, reg_map, instrs, seen_vars)
        elif arg.is_compound():
            name, arity = arg.value, len(arg.args)
            instrs.append(Instruction("put_structure", [(name, arity), ai]))
            for sub in arg.args:
                self._emit_query_set(sub, reg_map, instrs, seen_vars)

    def _emit_query_set_list(self, term, reg_map, instrs, seen_vars: set = None):
        """Emit set_* instructions for building a list in a query.

        For multi-element lists, emits intermediate put_list instructions
        to create nested list cells for each cdr.
        """
        if seen_vars is None:
            seen_vars = set()
        elements = []
        cur = term
        while cur.is_compound() and cur.value == "." and len(cur.args) == 2:
            elements.append(cur.args[0])
            cur = cur.args[1]
        tail = cur

        for i, elem in enumerate(elements):
            if i > 0:
                # Cdr of previous list cell → create inner list cell
                instrs.append(Instruction("put_list", [0]))
            self._emit_query_set(elem, reg_map, instrs, seen_vars)

        # Handle the tail
        if tail.is_atom() and tail.value == "[]":
            instrs.append(Instruction("set_constant", ["[]"]))
        elif tail.is_var():
            vn = reg_map.get(tail.value)
            if vn:
                if tail.value not in seen_vars:
                    seen_vars.add(tail.value)
                    instrs.append(Instruction("set_variable", [vn]))
                else:
                    instrs.append(Instruction("set_value", [vn]))
            else:
                instrs.append(Instruction("set_void", [1]))
        else:
            self._emit_query_set(tail, reg_map, instrs, seen_vars)

    def _emit_query_set(self, sub, reg_map, instrs, seen_vars: set = None):
        if seen_vars is None:
            seen_vars = set()
        if sub.is_var():
            vn = reg_map.get(sub.value)
            if vn is None:
                instrs.append(Instruction("set_void", [1]))
            elif sub.value not in seen_vars:
                seen_vars.add(sub.value)
                instrs.append(Instruction("set_variable", [vn]))
            else:
                instrs.append(Instruction("set_value", [vn]))
        elif sub.is_atom() or sub.is_num():
            instrs.append(Instruction("set_constant", [sub.value]))
        elif sub.is_compound() and sub.value == "." and len(sub.args) == 2:
            self._emit_query_set_list(sub, reg_map, instrs, seen_vars)
        elif sub.is_compound():
            name, arity = sub.value, len(sub.args)
            instrs.append(Instruction("set_void", [1]))  # simplified
        else:
            instrs.append(Instruction("set_void", [1]))


# ==========================================================================
# REPL
# ==========================================================================

def run_query(wam, compiler, query_text):
    """Run a query and print results."""
    wam.reset()
    try:
        result = compiler.compile_query(query_text)
        if not result:
            print("false.")
            return
        instrs, reg_map = result

        if not instrs:
            print("true.")
            return

        wam.code_labels["$query"] = instrs
        wam.current_label = "$query"
        wam.current_code = instrs
        wam.P = 0

        # Execute query setup instructions (put_*/set_*) until the first call so
        # that every query variable has been allocated on the heap.  Record the
        # stable heap address of each variable NOW – before any called clause can
        # overwrite the X registers that currently hold those addresses.
        while wam.P < len(instrs) and instrs[wam.P].opcode not in ("call", "execute"):
            wam.fetch_and_execute()

        var_heap_addrs: dict = {}
        for vname, vn in reg_map.items():
            if vn.startswith("X"):
                reg_num = int(vn[1:])
                if reg_num in wam.X:
                    var_heap_addrs[vname] = wam.X[reg_num]

        solutions = []

        while True:
            wam.halted = False
            wam.fail = False
            wam.run()

            if not wam.fail and wam.halted:
                sol = _capture_solution(wam, var_heap_addrs)
                solutions.append(sol)

            # Try backtracking regardless of success or failure
            if wam.B > 0:
                wam._backtrack()
                if wam.fail:
                    break
            else:
                break

        if not solutions:
            print("false.")
        else:
            for sol in solutions:
                if sol:
                    parts = [f"{k} = {_format_val(v)}" for k, v in sol.items()]
                    print(", ".join(parts) + ".")
                else:
                    print("true.")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def _capture_solution(wam, var_heap_addrs: dict) -> dict:
    """Display variable bindings using stable heap addresses captured before any call."""
    return {vname: wam._display(addr) for vname, addr in var_heap_addrs.items()}


def _format_val(v):
    if isinstance(v, list):
        return "[" + ", ".join(_format_val(x) for x in v) + "]"
    return str(v)


def repl():
    wam = WAM()
    compiler = Compiler(wam)

    print("WAM Prolog Interpreter")
    print("Commands:")
    print("  <clause>.          Load a clause (e.g., 'likes(john, mary).')")
    print("  ?- <query>.        Execute a query")
    print("  listing.           Show loaded predicates")
    print("  quit.              Exit")
    print()

    buf = ""
    while True:
        try:
            prompt = "| " if buf else ""
            line = input(prompt)
        except EOFError:
            break

        buf += " " + line

        if not buf.strip().endswith("."):
            continue

        text = buf.strip()
        buf = ""

        if text.lower() in ("quit.", "exit."):
            break

        text_no_dot = text.rstrip(".")

        if text.startswith("?-"):
            query_text = text[2:].strip().rstrip(".")
            if query_text:
                run_query(wam, compiler, query_text)
        elif text_no_dot.strip() == "listing":
            for key in sorted(wam.code_labels.keys()):
                if not key.startswith("$") and "_c" not in key:
                    print(f"{key}:")
                    for instr in wam.code_labels[key]:
                        print(f"  {instr}")
        else:
            try:
                compiler.load_program(text)
                print("Loaded.")
            except Exception as e:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="WAM Prolog Interpreter")
    ap.add_argument("file", nargs="?", help="Prolog source file")
    ap.add_argument("-q", "--query", help="Query to execute")
    args = ap.parse_args()

    wam = WAM()
    compiler = Compiler(wam)

    if args.file:
        with open(args.file) as f:
            compiler.load_program(f.read())

    if args.query:
        run_query(wam, compiler, args.query)
    elif not args.file:
        repl()


if __name__ == "__main__":
    main()
