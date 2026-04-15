"""Symbol table / atom interning for the Prolog-to-WASM compiler.

Assigns stable integer IDs to atoms and functors so the WASM runtime
can represent them as plain i32 values.

Encoding scheme (all fits in signed i32):

  - Numbers: stored as-is in the value field.  The CON tag on the
    heap cell distinguishes constants from other cell types at
    runtime, so the value field is free to hold either a number or
    an atom ID.

  - Atoms:  interned to sequential IDs starting at ATOM_BASE.
    ATOM_BASE = 0x4000_0000 (bit 30 set, bit 31 clear).
    This is well above any integer a Prolog program would use
    (up to ~1 billion), and leaves room for functor packing.

  - Functors:  packed as (atom_idx << 8) | arity where atom_idx is
    the *zero-based index* (not the ATOM_BASE offset).  This matches
    the existing convention in wam_wasm.py and keeps the packed form
    small enough to fit in i32.  With 24 bits for the index, we can
    represent up to 16M distinct functors.

Usage:
    syms = SymbolTable()
    syms.intern("tom")            # -> 0x40000000
    syms.intern("bob")            # -> 0x40000001
    syms.functor_pack("f", 2)     # -> (1 << 8) | 2 = 0x102
    syms.is_atom(0x40000000)      # -> True
    syms.atom_name(0x40000000)    # -> "tom"
"""

# Atom IDs: bit 30 set, bit 31 clear.  Leaves room for packed functors
# and stays within signed i32 range (max 0x7FFF_FFFF).
ATOM_BASE = 0x4000_0000


class SymbolTable:
    def __init__(self):
        self._atoms: dict[str, int] = {}       # name -> id (ATOM_BASE + idx)
        self._names: list[str] = []             # index -> name
        self._atom_counter: int = 0

    # -- atoms --

    def intern(self, name: str) -> int:
        """Return the integer ID for atom `name`, interning it if new."""
        if name in self._atoms:
            return self._atoms[name]
        idx = self._atom_counter
        self._atom_counter += 1
        self._names.append(name)
        atom_id = ATOM_BASE + idx
        self._atoms[name] = atom_id
        return atom_id

    def atom_index(self, name: str) -> int:
        """Return the zero-based index for an atom (without ATOM_BASE offset)."""
        self.intern(name)  # ensure it exists
        return self._atoms[name] - ATOM_BASE

    def is_atom(self, value: int) -> bool:
        """True if `value` is an interned atom ID."""
        return value >= ATOM_BASE

    def atom_name(self, atom_id: int) -> str:
        """Return the string name for an atom ID."""
        idx = atom_id - ATOM_BASE
        if 0 <= idx < len(self._names):
            return self._names[idx]
        raise KeyError(f"not an atom ID: {atom_id:#x}")

    def atom_count(self) -> int:
        return self._atom_counter

    # -- functors --

    def functor_pack(self, name: str, arity: int) -> int:
        """Return the packed i32 for functor name/arity.

        Layout: (atom_index << 8) | arity
        Arity must fit in 8 bits (0..255).
        The atom_index is the zero-based interning index, NOT the
        ATOM_BASE offset — this keeps the packed value small.
        """
        idx = self.atom_index(name)
        assert 0 <= arity <= 255, f"arity {arity} too large"
        return (idx << 8) | arity

    def functor_unpack(self, packed: int) -> tuple[str, int]:
        """Decode a packed functor back to (name, arity)."""
        arity = packed & 0xFF
        idx = packed >> 8
        return self._names[idx], arity

    # -- constants (used by get_constant / put_constant) --

    def encode_constant(self, value) -> int:
        """Encode a constant value (str atom or int/float number) to i32.

        Strings (atoms) are interned.  Numbers are stored directly.
        """
        if isinstance(value, str):
            return self.intern(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # For the PoC, truncate floats to int.
            # A full implementation would use a separate float heap.
            return int(value)
        raise TypeError(f"cannot encode constant: {value!r}")

    def decode_constant(self, value: int):
        """Decode an i32 constant back to its Python value."""
        if self.is_atom(value):
            return self.atom_name(value)
        return value

    # -- bulk: intern all atoms/functors from a compiled program --

    # WAM instructions that take a functor (name, arity) tuple
    _FUNCTOR_OPCODES = frozenset({
        "get_structure", "put_structure",
    })

    # WAM instructions that take a string constant
    _CONSTANT_OPCODES = frozenset({
        "get_constant", "put_constant",
        "unify_constant", "set_constant",
    })

    def intern_program(self, predicates: dict, queries: list):
        """Walk all compiled predicates and queries, interning every
        atom and functor name encountered.

        Call this before emitting WASM so all IDs are stable.
        """
        for key, pred in predicates.items():
            # key is like "parent/2" — intern the predicate name
            name = key.rsplit("/", 1)[0]
            self.intern(name)

            for _label, instrs in pred.clauses:
                self._intern_instrs(instrs)

        for instrs, _reg_map in queries:
            self._intern_instrs(instrs)

    def _intern_instrs(self, instrs):
        for instr in instrs:
            if instr.opcode in self._FUNCTOR_OPCODES:
                # First arg is (name, arity)
                name, arity = instr.args[0]
                self.functor_pack(name, arity)
            elif instr.opcode in self._CONSTANT_OPCODES:
                # First arg is a string atom name
                val = instr.args[0]
                if isinstance(val, str):
                    self.intern(val)

    # -- summary --

    def summary(self) -> str:
        lines = [f"Symbol table: {self._atom_counter} atoms"]
        for i, name in enumerate(self._names):
            aid = ATOM_BASE + i
            packed_example = ""
            lines.append(f"  {aid:#010x}  idx={i:<4d}  {name}")
        return "\n".join(lines)
