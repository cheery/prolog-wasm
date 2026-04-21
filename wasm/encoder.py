"""WebAssembly 3.0 binary format encoder.

Composable combinators for encoding .wasm binary structures.
Every combinator returns `bytes`; they compose through concatenation.

    result = section(1, vec(types, functype))
    result = byte(0x00) + u32(42) + name("hello")
"""

import struct


# ===================================================================
# LEB128 variable-length integer encoding
# ===================================================================

def leb128_u(value: int) -> bytes:
    """Encode an unsigned integer as unsigned LEB128."""
    if value < 0:
        raise ValueError(f"unsigned LEB128 requires value >= 0, got {value}")
    parts = []
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            parts.append(b | 0x80)
        else:
            parts.append(b)
            break
    return bytes(parts)


def leb128_s(value: int) -> bytes:
    """Encode a signed integer as signed LEB128."""
    parts = []
    more = True
    while more:
        b = value & 0x7F
        value >>= 7
        more = not (((value == 0) and (b & 0x40 == 0))
                     or ((value == -1) and (b & 0x40 != 0)))
        if more:
            b |= 0x80
        parts.append(b)
    return bytes(parts)


# ===================================================================
# Primitive combinators
# ===================================================================

def byte(v: int) -> bytes:
    """Encode a single byte."""
    return bytes([v & 0xFF])


def u32(v: int) -> bytes:
    """Encode an unsigned 32-bit integer (LEB128)."""
    return leb128_u(v)


def s33(v: int) -> bytes:
    """Encode a signed 33-bit integer (LEB128)."""
    return leb128_s(v)


def u64(v: int) -> bytes:
    """Encode an unsigned 64-bit integer (LEB128)."""
    return leb128_u(v)


def i32(v: int) -> bytes:
    """Encode a signed 32-bit integer (LEB128)."""
    return leb128_s(v)


def i64(v: int) -> bytes:
    """Encode a signed 64-bit integer (LEB128)."""
    return leb128_s(v)


def f32(v: float) -> bytes:
    """Encode an IEEE 754 32-bit float (little-endian)."""
    return struct.pack('<f', v)


def f64(v: float) -> bytes:
    """Encode an IEEE 754 64-bit float (little-endian)."""
    return struct.pack('<d', v)


def f32_bits(v: int) -> bytes:
    """Encode a 32-bit float from its raw IEEE 754 bit pattern."""
    return struct.pack('<I', v)


def f64_bits(v: int) -> bytes:
    """Encode a 64-bit float from its raw IEEE 754 bit pattern."""
    return struct.pack('<Q', v)


def name(s: str) -> bytes:
    """Encode a name: length-prefixed UTF-8 string."""
    encoded = s.encode('utf-8')
    return u32(len(encoded)) + encoded


def vec(items, encode_fn):
    """Encode a vector: count prefix followed by encoded items.

    encode_fn is called with each item and must return bytes.
    """
    return u32(len(items)) + b''.join(encode_fn(item) for item in items)


def section(section_id: int, payload: bytes) -> bytes:
    """Wrap payload in a section: id + size + data."""
    return byte(section_id) + u32(len(payload)) + payload


def raw(data: bytes) -> bytes:
    """Pass-through raw bytes."""
    return data


# ===================================================================
# Type constants
# ===================================================================

# Value types
I32 = 0x7F
I64 = 0x7E
F32 = 0x7D
F64 = 0x7C
V128 = 0x7B

# Heap types (absolute)
HT_EXN = 0x69
HT_ARRAY = 0x6A
HT_STRUCT = 0x6B
HT_I31 = 0x6C
HT_EQ = 0x6D
HT_ANY = 0x6E
HT_EXTERN = 0x6F
HT_FUNC = 0x70
HT_NONE = 0x71
HT_NOEXTERN = 0x72
HT_NOFUNC = 0x73
HT_NOEXN = 0x74

# Pack types
I8 = 0x78
I16 = 0x77

# Common shorthand ref types (same as abs heap types)
FUNCREF = HT_FUNC       # (ref null func)
EXTERNREF = HT_EXTERN   # (ref null extern)

# External type kinds
EXT_FUNC = 0x00
EXT_TABLE = 0x01
EXT_MEM = 0x02
EXT_GLOBAL = 0x03
EXT_TAG = 0x04

# Section IDs
SEC_CUSTOM = 0
SEC_TYPE = 1
SEC_IMPORT = 2
SEC_FUNC = 3
SEC_TABLE = 4
SEC_MEMORY = 5
SEC_GLOBAL = 6
SEC_EXPORT = 7
SEC_START = 8
SEC_ELEM = 9
SEC_CODE = 10
SEC_DATA = 11
SEC_DATACOUNT = 12
SEC_TAG = 13

# Module header
WASM_MAGIC = b'\x00asm'
WASM_VERSION = b'\x01\x00\x00\x00'


# ===================================================================
# Type encoders
# ===================================================================

def valtype(vt) -> bytes:
    """Encode a value type (I32, I64, F32, F64, V128, or ref type bytes)."""
    if isinstance(vt, bytes):
        return vt
    return byte(vt)


def heaptype(ht) -> bytes:
    """Encode a heap type.

    ht can be an absolute heap type constant (HT_FUNC etc.)
    or a non-negative int for a type index.
    """
    if isinstance(ht, bytes):
        return ht
    if isinstance(ht, int) and ht >= HT_EXN:
        return byte(ht)
    return s33(ht)


def reftype(nullable: bool, ht) -> bytes:
    """Encode a reference type.

    nullable: True for (ref null ht), False for (ref ht)
    ht: heap type (absolute constant or type index)

    Uses the shorthand encoding for nullable + abs heap types.
    """
    ht_enc = heaptype(ht)
    if nullable and isinstance(ht, int) and ht >= HT_EXN:
        return ht_enc  # shorthand: just the abs heap type byte
    return byte(0x63 if nullable else 0x64) + ht_enc


def functype(params, results) -> bytes:
    """Encode a function type: 0x60 + param_types + result_types.

    params:  list of value types
    results: list of value types
    """
    return byte(0x60) + vec(params, valtype) + vec(results, valtype)


def storagetype(st) -> bytes:
    """Encode a storage type (value type or pack type)."""
    return byte(st)


def fieldtype(st, mutable: bool = False) -> bytes:
    """Encode a field type: storage_type + mutability."""
    return storagetype(st) + byte(0x01 if mutable else 0x00)


def comptype_func(params, results) -> bytes:
    """Encode a func composite type."""
    return functype(params, results)


def comptype_struct(fields) -> bytes:
    """Encode a struct composite type: 0x5F + field_types."""
    return byte(0x5F) + vec(fields, lambda f: fieldtype(f[0], f[1]) if isinstance(f, tuple) else fieldtype(f, False))


def comptype_array(field) -> bytes:
    """Encode an array composite type: 0x5E + field_type."""
    return byte(0x5E) + (fieldtype(field[0], field[1]) if isinstance(field, tuple) else fieldtype(field, False))


def subtype(comptype_bytes, final: bool = True, supers=()) -> bytes:
    """Encode a sub type.

    comptype_bytes: encoded composite type (from comptype_*)
    final:  whether this is a final type
    supers: list of super type indices
    """
    if not supers:
        if final:
            return comptype_bytes  # implicit SUB FINAL eps
        else:
            return byte(0x50) + vec(supers, u32) + comptype_bytes
    if final:
        return byte(0x4F) + vec(supers, u32) + comptype_bytes
    return byte(0x50) + vec(supers, u32) + comptype_bytes


def rectype(subtypes_bytes) -> bytes:
    """Encode a rec type wrapping multiple sub types.

    subtypes_bytes: list of encoded sub types
    For a single subtype, just use the subtype encoding directly.
    """
    if len(subtypes_bytes) == 1:
        return subtypes_bytes[0]
    return byte(0x4E) + vec(subtypes_bytes, raw)


def limits(min_val: int, max_val=None, addr64: bool = False) -> bytes:
    """Encode limits.

    min_val: minimum value
    max_val: optional maximum value
    addr64:  use I64 address type
    """
    if addr64:
        prefix = 0x05 if max_val is not None else 0x04
    else:
        prefix = 0x01 if max_val is not None else 0x00
    result = byte(prefix) + u64(min_val)
    if max_val is not None:
        result += u64(max_val)
    return result


def globaltype(vt, mutable: bool = False) -> bytes:
    """Encode a global type: value_type + mutability."""
    return valtype(vt) + byte(0x01 if mutable else 0x00)


def memtype(min_val: int, max_val=None, addr64: bool = False) -> bytes:
    """Encode a memory type: limits."""
    return limits(min_val, max_val, addr64)


def tabletype(ref_null, ht, min_val: int, max_val=None, addr64: bool = False) -> bytes:
    """Encode a table type: ref_type + limits.

    ref_null: True for nullable ref, False for non-nullable
    ht:       heap type
    """
    return reftype(ref_null, ht) + limits(min_val, max_val, addr64)


def tagtype(typeidx: int) -> bytes:
    """Encode a tag type: 0x00 + type_index."""
    return byte(0x00) + u32(typeidx)


def externtype_func(typeidx: int) -> bytes:
    """Encode a func external type."""
    return byte(EXT_FUNC) + u32(typeidx)


def externtype_table(ref_null, ht, min_val, max_val=None, addr64=False) -> bytes:
    """Encode a table external type."""
    return byte(EXT_TABLE) + tabletype(ref_null, ht, min_val, max_val, addr64)


def externtype_mem(min_val, max_val=None, addr64=False) -> bytes:
    """Encode a memory external type."""
    return byte(EXT_MEM) + memtype(min_val, max_val, addr64)


def externtype_global(vt, mutable=False) -> bytes:
    """Encode a global external type."""
    return byte(EXT_GLOBAL) + globaltype(vt, mutable)


def externtype_tag(typeidx: int) -> bytes:
    """Encode a tag external type."""
    return byte(EXT_TAG) + tagtype(typeidx)


def blocktype(bt) -> bytes:
    """Encode a block type.

    bt: None for void, a valtype constant for single result,
        or a non-negative int for a type index.
    """
    if bt is None:
        return byte(0x40)
    if isinstance(bt, int):
        if bt >= V128:
            return byte(bt)
        return s33(bt)
    return bt


# ===================================================================
# Instruction encoders
# ===================================================================

# -- Helpers for extended opcodes --

def _prefix_fb(subop: int, *parts: bytes) -> bytes:
    """Encode a 0xFB-prefixed instruction (GC/extended ref)."""
    return byte(0xFB) + u32(subop) + b''.join(parts)


def _prefix_fc(subop: int, *parts: bytes) -> bytes:
    """Encode a 0xFC-prefixed instruction (miscellaneous)."""
    return byte(0xFC) + u32(subop) + b''.join(parts)


def _prefix_fd(subop: int, *parts: bytes) -> bytes:
    """Encode a 0xFD-prefixed instruction (SIMD)."""
    return byte(0xFD) + u32(subop) + b''.join(parts)


# -- Parametric instructions --

def unreachable() -> bytes:
    return byte(0x00)

def nop() -> bytes:
    return byte(0x01)

def drop() -> bytes:
    return byte(0x1A)

def select() -> bytes:
    return byte(0x1B)

def select_t(*types) -> bytes:
    """Typed select: select with explicit value types."""
    return byte(0x1C) + vec(types, valtype)


# -- Control instructions --

def block(bt, instrs) -> bytes:
    """Encode a block: blocktype + instructions + end."""
    return byte(0x02) + blocktype(bt) + b''.join(instrs) + byte(0x0B)

def loop(bt, instrs) -> bytes:
    """Encode a loop: blocktype + instructions + end."""
    return byte(0x03) + blocktype(bt) + b''.join(instrs) + byte(0x0B)

def if_(bt, then_instrs, else_instrs=None) -> bytes:
    """Encode an if/else: blocktype + then_instrs + [else + else_instrs] + end."""
    result = byte(0x04) + blocktype(bt) + b''.join(then_instrs)
    if else_instrs:
        result += byte(0x05) + b''.join(else_instrs)
    result += byte(0x0B)
    return result

def br(label: int) -> bytes:
    return byte(0x0C) + u32(label)

def br_if(label: int) -> bytes:
    return byte(0x0D) + u32(label)

def br_table(labels, default: int) -> bytes:
    """Encode br_table: count + labels + default."""
    return byte(0x0E) + vec(labels, u32) + u32(default)

def return_() -> bytes:
    return byte(0x0F)

def call(funcidx: int) -> bytes:
    return byte(0x10) + u32(funcidx)

def call_indirect(tableidx: int, typeidx: int) -> bytes:
    return byte(0x11) + u32(typeidx) + u32(tableidx)

def return_call(funcidx: int) -> bytes:
    return byte(0x12) + u32(funcidx)

def return_call_indirect(tableidx: int, typeidx: int) -> bytes:
    return byte(0x13) + u32(typeidx) + u32(tableidx)

def call_ref(typeidx: int) -> bytes:
    return byte(0x14) + u32(typeidx)

def return_call_ref(typeidx: int) -> bytes:
    return byte(0x15) + u32(typeidx)

def throw(tagidx: int) -> bytes:
    return byte(0x08) + u32(tagidx)

def throw_ref() -> bytes:
    return byte(0x0A)

def try_table(bt, catches, instrs) -> bytes:
    """Encode try_table: blocktype + catch_clauses + instructions + end.

    catches: list of encoded catch clauses (use catch/catch_ref/catch_all/catch_all_ref)
    """
    return byte(0x1F) + blocktype(bt) + vec(catches, raw) + b''.join(instrs) + byte(0x0B)

# Catch clause builders for try_table
def catch_clause(tagidx: int, label: int) -> bytes:
    return byte(0x00) + u32(tagidx) + u32(label)

def catch_ref(tagidx: int, label: int) -> bytes:
    return byte(0x01) + u32(tagidx) + u32(label)

def catch_all(label: int) -> bytes:
    return byte(0x02) + u32(label)

def catch_all_ref(label: int) -> bytes:
    return byte(0x03) + u32(label)

def br_on_null(label: int) -> bytes:
    return byte(0xD5) + u32(label)

def br_on_non_null(label: int) -> bytes:
    return byte(0xD6) + u32(label)

def br_on_cast(label, ht1, ht2, null1=False, null2=False) -> bytes:
    """Encode br_on_cast."""
    flags = (0x01 if null1 else 0x00) | (0x02 if null2 else 0x00)
    return _prefix_fb(24, u32(flags), u32(label), heaptype(ht1), heaptype(ht2))

def br_on_cast_fail(label, ht1, ht2, null1=False, null2=False) -> bytes:
    """Encode br_on_cast_fail."""
    flags = (0x01 if null1 else 0x00) | (0x02 if null2 else 0x00)
    return _prefix_fb(25, u32(flags), u32(label), heaptype(ht1), heaptype(ht2))

def end() -> bytes:
    return byte(0x0B)


# -- Variable instructions --

def local_get(idx: int) -> bytes:
    return byte(0x20) + u32(idx)

def local_set(idx: int) -> bytes:
    return byte(0x21) + u32(idx)

def local_tee(idx: int) -> bytes:
    return byte(0x22) + u32(idx)

def global_get(idx: int) -> bytes:
    return byte(0x23) + u32(idx)

def global_set(idx: int) -> bytes:
    return byte(0x24) + u32(idx)


# -- Table instructions --

def table_get(idx: int) -> bytes:
    return byte(0x25) + u32(idx)

def table_set(idx: int) -> bytes:
    return byte(0x26) + u32(idx)

def table_init(tableidx: int, elemidx: int) -> bytes:
    return _prefix_fc(12, u32(elemidx), u32(tableidx))

def elem_drop(idx: int) -> bytes:
    return _prefix_fc(13, u32(idx))

def table_copy(dst: int, src: int) -> bytes:
    return _prefix_fc(14, u32(dst), u32(src))

def table_grow(idx: int) -> bytes:
    return _prefix_fc(15, u32(idx))

def table_size(idx: int) -> bytes:
    return _prefix_fc(16, u32(idx))

def table_fill(idx: int) -> bytes:
    return _prefix_fc(17, u32(idx))


# -- Memory instructions --
# Each takes optional offset and alignment (natural alignment default).

def _memarg(align: int, offset: int) -> bytes:
    """Encode a memory operand for memory 0."""
    return u32(align) + u32(offset)

def i32_load(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x28) + _memarg(align, offset)

def i64_load(offset: int = 0, align: int = 3) -> bytes:
    return byte(0x29) + _memarg(align, offset)

def f32_load(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x2A) + _memarg(align, offset)

def f64_load(offset: int = 0, align: int = 3) -> bytes:
    return byte(0x2B) + _memarg(align, offset)

def i32_load8_s(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x2C) + _memarg(align, offset)

def i32_load8_u(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x2D) + _memarg(align, offset)

def i32_load16_s(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x2E) + _memarg(align, offset)

def i32_load16_u(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x2F) + _memarg(align, offset)

def i64_load8_s(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x30) + _memarg(align, offset)

def i64_load8_u(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x31) + _memarg(align, offset)

def i64_load16_s(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x32) + _memarg(align, offset)

def i64_load16_u(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x33) + _memarg(align, offset)

def i64_load32_s(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x34) + _memarg(align, offset)

def i64_load32_u(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x35) + _memarg(align, offset)

def i32_store(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x36) + _memarg(align, offset)

def i64_store(offset: int = 0, align: int = 3) -> bytes:
    return byte(0x37) + _memarg(align, offset)

def f32_store(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x38) + _memarg(align, offset)

def f64_store(offset: int = 0, align: int = 3) -> bytes:
    return byte(0x39) + _memarg(align, offset)

def i32_store8(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x3A) + _memarg(align, offset)

def i32_store16(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x3B) + _memarg(align, offset)

def i64_store8(offset: int = 0, align: int = 0) -> bytes:
    return byte(0x3C) + _memarg(align, offset)

def i64_store16(offset: int = 0, align: int = 1) -> bytes:
    return byte(0x3D) + _memarg(align, offset)

def i64_store32(offset: int = 0, align: int = 2) -> bytes:
    return byte(0x3E) + _memarg(align, offset)

def memory_size(memidx: int = 0) -> bytes:
    return byte(0x3F) + u32(memidx)

def memory_grow(memidx: int = 0) -> bytes:
    return byte(0x40) + u32(memidx)

def memory_init(memidx: int, dataidx: int) -> bytes:
    return _prefix_fc(8, u32(dataidx), u32(memidx))

def data_drop(idx: int) -> bytes:
    return _prefix_fc(9, u32(idx))

def memory_copy(dst: int, src: int) -> bytes:
    return _prefix_fc(10, u32(dst), u32(src))

def memory_fill(memidx: int) -> bytes:
    return _prefix_fc(11, u32(memidx))


# -- Reference instructions --

def ref_null(ht) -> bytes:
    """ref.null heap_type"""
    return byte(0xD0) + heaptype(ht)

def ref_is_null() -> bytes:
    return byte(0xD1)

def ref_func(idx: int) -> bytes:
    return byte(0xD2) + u32(idx)

def ref_eq() -> bytes:
    return byte(0xD3)

def ref_as_non_null() -> bytes:
    return byte(0xD4)

def ref_test(ht) -> bytes:
    """ref.test (ref ht) - non-nullable."""
    return _prefix_fb(20, heaptype(ht))

def ref_test_null(ht) -> bytes:
    """ref.test (ref null ht) - nullable."""
    return _prefix_fb(21, heaptype(ht))

def ref_cast(ht) -> bytes:
    """ref.cast (ref ht) - non-nullable."""
    return _prefix_fb(22, heaptype(ht))

def ref_cast_null(ht) -> bytes:
    """ref.cast (ref null ht) - nullable."""
    return _prefix_fb(23, heaptype(ht))


# -- GC: struct instructions --

def struct_new(typeidx: int) -> bytes:
    return _prefix_fb(0, u32(typeidx))

def struct_new_default(typeidx: int) -> bytes:
    return _prefix_fb(1, u32(typeidx))

def struct_get(typeidx: int, fieldidx: int) -> bytes:
    return _prefix_fb(2, u32(typeidx), u32(fieldidx))

def struct_get_s(typeidx: int, fieldidx: int) -> bytes:
    return _prefix_fb(3, u32(typeidx), u32(fieldidx))

def struct_get_u(typeidx: int, fieldidx: int) -> bytes:
    return _prefix_fb(4, u32(typeidx), u32(fieldidx))

def struct_set(typeidx: int, fieldidx: int) -> bytes:
    return _prefix_fb(5, u32(typeidx), u32(fieldidx))


# -- GC: array instructions --

def array_new(typeidx: int) -> bytes:
    return _prefix_fb(6, u32(typeidx))

def array_new_default(typeidx: int) -> bytes:
    return _prefix_fb(7, u32(typeidx))

def array_new_fixed(typeidx: int, n: int) -> bytes:
    return _prefix_fb(8, u32(typeidx), u32(n))

def array_new_data(typeidx: int, dataidx: int) -> bytes:
    return _prefix_fb(9, u32(typeidx), u32(dataidx))

def array_new_elem(typeidx: int, elemidx: int) -> bytes:
    return _prefix_fb(10, u32(typeidx), u32(elemidx))

def array_get(typeidx: int) -> bytes:
    return _prefix_fb(11, u32(typeidx))

def array_get_s(typeidx: int) -> bytes:
    return _prefix_fb(12, u32(typeidx))

def array_get_u(typeidx: int) -> bytes:
    return _prefix_fb(13, u32(typeidx))

def array_set(typeidx: int) -> bytes:
    return _prefix_fb(14, u32(typeidx))

def array_len() -> bytes:
    return _prefix_fb(15)

def array_fill(typeidx: int) -> bytes:
    return _prefix_fb(16, u32(typeidx))

def array_copy(dst_typeidx: int, src_typeidx: int) -> bytes:
    return _prefix_fb(17, u32(dst_typeidx), u32(src_typeidx))

def array_init_data(typeidx: int, dataidx: int) -> bytes:
    return _prefix_fb(18, u32(typeidx), u32(dataidx))

def array_init_elem(typeidx: int, elemidx: int) -> bytes:
    return _prefix_fb(19, u32(typeidx), u32(elemidx))


# -- GC: i31 / extern instructions --

def ref_i31() -> bytes:
    return _prefix_fb(28)

def i31_get_s() -> bytes:
    return _prefix_fb(29)

def i31_get_u() -> bytes:
    return _prefix_fb(30)

def any_convert_extern() -> bytes:
    return _prefix_fb(26)

def extern_convert_any() -> bytes:
    return _prefix_fb(27)


# -- Numeric constant instructions --

def i32_const(v: int) -> bytes:
    return byte(0x41) + i32(v)

def i64_const(v: int) -> bytes:
    return byte(0x42) + i64(v)

def f32_const(v: float) -> bytes:
    return byte(0x43) + f32(v)

def f64_const(v: float) -> bytes:
    return byte(0x44) + f64(v)


# -- Numeric: i32 operations --

def i32_eqz() -> bytes: return byte(0x45)
def i32_eq() -> bytes:  return byte(0x46)
def i32_ne() -> bytes:  return byte(0x47)
def i32_lt_s() -> bytes: return byte(0x48)
def i32_lt_u() -> bytes: return byte(0x49)
def i32_gt_s() -> bytes: return byte(0x4A)
def i32_gt_u() -> bytes: return byte(0x4B)
def i32_le_s() -> bytes: return byte(0x4C)
def i32_le_u() -> bytes: return byte(0x4D)
def i32_ge_s() -> bytes: return byte(0x4E)
def i32_ge_u() -> bytes: return byte(0x4F)

def i32_clz() -> bytes: return byte(0x67)
def i32_ctz() -> bytes: return byte(0x68)
def i32_popcnt() -> bytes: return byte(0x69)

def i32_add() -> bytes: return byte(0x6A)
def i32_sub() -> bytes: return byte(0x6B)
def i32_mul() -> bytes: return byte(0x6C)
def i32_div_s() -> bytes: return byte(0x6D)
def i32_div_u() -> bytes: return byte(0x6E)
def i32_rem_s() -> bytes: return byte(0x6F)
def i32_rem_u() -> bytes: return byte(0x70)
def i32_and() -> bytes: return byte(0x71)
def i32_or() -> bytes:  return byte(0x72)
def i32_xor() -> bytes: return byte(0x73)
def i32_shl() -> bytes: return byte(0x74)
def i32_shr_s() -> bytes: return byte(0x75)
def i32_shr_u() -> bytes: return byte(0x76)
def i32_rotl() -> bytes: return byte(0x77)
def i32_rotr() -> bytes: return byte(0x78)

def i32_extend8_s() -> bytes: return byte(0xC0)
def i32_extend16_s() -> bytes: return byte(0xC1)


# -- Numeric: i64 operations --

def i64_eqz() -> bytes: return byte(0x50)
def i64_eq() -> bytes:  return byte(0x51)
def i64_ne() -> bytes:  return byte(0x52)
def i64_lt_s() -> bytes: return byte(0x53)
def i64_lt_u() -> bytes: return byte(0x54)
def i64_gt_s() -> bytes: return byte(0x55)
def i64_gt_u() -> bytes: return byte(0x56)
def i64_le_s() -> bytes: return byte(0x57)
def i64_le_u() -> bytes: return byte(0x58)
def i64_ge_s() -> bytes: return byte(0x59)
def i64_ge_u() -> bytes: return byte(0x5A)

def i64_clz() -> bytes: return byte(0x79)
def i64_ctz() -> bytes: return byte(0x7A)
def i64_popcnt() -> bytes: return byte(0x7B)

def i64_add() -> bytes: return byte(0x7C)
def i64_sub() -> bytes: return byte(0x7D)
def i64_mul() -> bytes: return byte(0x7E)
def i64_div_s() -> bytes: return byte(0x7F)
def i64_div_u() -> bytes: return byte(0x80)
def i64_rem_s() -> bytes: return byte(0x81)
def i64_rem_u() -> bytes: return byte(0x82)
def i64_and() -> bytes: return byte(0x83)
def i64_or() -> bytes:  return byte(0x84)
def i64_xor() -> bytes: return byte(0x85)
def i64_shl() -> bytes: return byte(0x86)
def i64_shr_s() -> bytes: return byte(0x87)
def i64_shr_u() -> bytes: return byte(0x88)
def i64_rotl() -> bytes: return byte(0x89)
def i64_rotr() -> bytes: return byte(0x8A)

def i64_extend8_s() -> bytes: return byte(0xC2)
def i64_extend16_s() -> bytes: return byte(0xC3)
def i64_extend32_s() -> bytes: return byte(0xC4)


# -- Numeric: f32 operations --

def f32_abs() -> bytes: return byte(0x8B)
def f32_neg() -> bytes: return byte(0x8C)
def f32_ceil() -> bytes: return byte(0x8D)
def f32_floor() -> bytes: return byte(0x8E)
def f32_trunc() -> bytes: return byte(0x8F)
def f32_nearest() -> bytes: return byte(0x90)
def f32_sqrt() -> bytes: return byte(0x91)

def f32_add() -> bytes: return byte(0x92)
def f32_sub() -> bytes: return byte(0x93)
def f32_mul() -> bytes: return byte(0x94)
def f32_div() -> bytes: return byte(0x95)
def f32_min() -> bytes: return byte(0x96)
def f32_max() -> bytes: return byte(0x97)
def f32_copysign() -> bytes: return byte(0x98)

def f32_eq() -> bytes: return byte(0x5B)
def f32_ne() -> bytes: return byte(0x5C)
def f32_lt() -> bytes: return byte(0x5D)
def f32_gt() -> bytes: return byte(0x5E)
def f32_le() -> bytes: return byte(0x5F)
def f32_ge() -> bytes: return byte(0x60)


# -- Numeric: f64 operations --

def f64_abs() -> bytes: return byte(0x99)
def f64_neg() -> bytes: return byte(0x9A)
def f64_ceil() -> bytes: return byte(0x9B)
def f64_floor() -> bytes: return byte(0x9C)
def f64_trunc() -> bytes: return byte(0x9D)
def f64_nearest() -> bytes: return byte(0x9E)
def f64_sqrt() -> bytes: return byte(0x9F)

def f64_add() -> bytes: return byte(0xA0)
def f64_sub() -> bytes: return byte(0xA1)
def f64_mul() -> bytes: return byte(0xA2)
def f64_div() -> bytes: return byte(0xA3)
def f64_min() -> bytes: return byte(0xA4)
def f64_max() -> bytes: return byte(0xA5)
def f64_copysign() -> bytes: return byte(0xA6)

def f64_eq() -> bytes: return byte(0x61)
def f64_ne() -> bytes: return byte(0x62)
def f64_lt() -> bytes: return byte(0x63)
def f64_gt() -> bytes: return byte(0x64)
def f64_le() -> bytes: return byte(0x65)
def f64_ge() -> bytes: return byte(0x66)


# -- Numeric: conversion operations --

def i32_wrap_i64() -> bytes: return byte(0xA7)
def i32_trunc_f32_s() -> bytes: return byte(0xA8)
def i32_trunc_f32_u() -> bytes: return byte(0xA9)
def i32_trunc_f64_s() -> bytes: return byte(0xAA)
def i32_trunc_f64_u() -> bytes: return byte(0xAB)
def i64_extend_i32_s() -> bytes: return byte(0xAC)
def i64_extend_i32_u() -> bytes: return byte(0xAD)
def i64_trunc_f32_s() -> bytes: return byte(0xAE)
def i64_trunc_f32_u() -> bytes: return byte(0xAF)
def i64_trunc_f64_s() -> bytes: return byte(0xB0)
def i64_trunc_f64_u() -> bytes: return byte(0xB1)
def f32_convert_i32_s() -> bytes: return byte(0xB2)
def f32_convert_i32_u() -> bytes: return byte(0xB3)
def f32_convert_i64_s() -> bytes: return byte(0xB4)
def f32_convert_i64_u() -> bytes: return byte(0xB5)
def f32_demote_f64() -> bytes: return byte(0xB6)
def f64_convert_i32_s() -> bytes: return byte(0xB7)
def f64_convert_i32_u() -> bytes: return byte(0xB8)
def f64_convert_i64_s() -> bytes: return byte(0xB9)
def f64_convert_i64_u() -> bytes: return byte(0xBA)
def f64_promote_f32() -> bytes: return byte(0xBB)
def i32_reinterpret_f32() -> bytes: return byte(0xBC)
def i64_reinterpret_f64() -> bytes: return byte(0xBD)
def f32_reinterpret_i32() -> bytes: return byte(0xBE)
def f64_reinterpret_i64() -> bytes: return byte(0xBF)


# -- Numeric: saturating truncation --

def i32_trunc_sat_f32_s() -> bytes: return _prefix_fc(0)
def i32_trunc_sat_f32_u() -> bytes: return _prefix_fc(1)
def i32_trunc_sat_f64_s() -> bytes: return _prefix_fc(2)
def i32_trunc_sat_f64_u() -> bytes: return _prefix_fc(3)
def i64_trunc_sat_f32_s() -> bytes: return _prefix_fc(4)
def i64_trunc_sat_f32_u() -> bytes: return _prefix_fc(5)
def i64_trunc_sat_f64_s() -> bytes: return _prefix_fc(6)
def i64_trunc_sat_f64_u() -> bytes: return _prefix_fc(7)


# -- Vector (SIMD) instructions --

def v128_load(offset=0, align=4) -> bytes:
    return _prefix_fd(0, _memarg(align, offset))

def v128_store(offset=0, align=4) -> bytes:
    return _prefix_fd(11, _memarg(align, offset))

def v128_const(b: bytes) -> bytes:
    """v128.const from 16 raw bytes."""
    assert len(b) == 16
    return _prefix_fd(12, b)

def i8x16_shuffle(lanes) -> bytes:
    """i8x16.shuffle with 16 lane indices (each 0..31)."""
    assert len(lanes) == 16
    return _prefix_fd(13, bytes(lanes))

def i8x16_swizzle() -> bytes:
    return _prefix_fd(14)

def i8x16_splat() -> bytes:
    return _prefix_fd(15)

def i16x8_splat() -> bytes:
    return _prefix_fd(16)

def i32x4_splat() -> bytes:
    return _prefix_fd(17)

def i64x2_splat() -> bytes:
    return _prefix_fd(18)

def f32x4_splat() -> bytes:
    return _prefix_fd(19)

def f64x2_splat() -> bytes:
    return _prefix_fd(20)

# Vector extract/replace lane
def i8x16_extract_lane_s(lane) -> bytes: return _prefix_fd(21, byte(lane))
def i8x16_extract_lane_u(lane) -> bytes: return _prefix_fd(22, byte(lane))
def i8x16_replace_lane(lane) -> bytes: return _prefix_fd(23, byte(lane))
def i16x8_extract_lane_s(lane) -> bytes: return _prefix_fd(24, byte(lane))
def i16x8_extract_lane_u(lane) -> bytes: return _prefix_fd(25, byte(lane))
def i16x8_replace_lane(lane) -> bytes: return _prefix_fd(26, byte(lane))
def i32x4_extract_lane(lane) -> bytes: return _prefix_fd(27, byte(lane))
def i32x4_replace_lane(lane) -> bytes: return _prefix_fd(28, byte(lane))
def i64x2_extract_lane(lane) -> bytes: return _prefix_fd(29, byte(lane))
def i64x2_replace_lane(lane) -> bytes: return _prefix_fd(30, byte(lane))
def f32x4_extract_lane(lane) -> bytes: return _prefix_fd(31, byte(lane))
def f32x4_replace_lane(lane) -> bytes: return _prefix_fd(32, byte(lane))
def f64x2_extract_lane(lane) -> bytes: return _prefix_fd(33, byte(lane))
def f64x2_replace_lane(lane) -> bytes: return _prefix_fd(34, byte(lane))

# Vector comparisons
def i8x16_eq() -> bytes: return _prefix_fd(35)
def i8x16_ne() -> bytes: return _prefix_fd(36)
def i8x16_lt_s() -> bytes: return _prefix_fd(37)
def i8x16_lt_u() -> bytes: return _prefix_fd(38)
def i8x16_gt_s() -> bytes: return _prefix_fd(39)
def i8x16_gt_u() -> bytes: return _prefix_fd(40)
def i8x16_le_s() -> bytes: return _prefix_fd(41)
def i8x16_le_u() -> bytes: return _prefix_fd(42)
def i8x16_ge_s() -> bytes: return _prefix_fd(43)
def i8x16_ge_u() -> bytes: return _prefix_fd(44)

def i16x8_eq() -> bytes: return _prefix_fd(45)
def i16x8_ne() -> bytes: return _prefix_fd(46)
def i16x8_lt_s() -> bytes: return _prefix_fd(47)
def i16x8_lt_u() -> bytes: return _prefix_fd(48)
def i16x8_gt_s() -> bytes: return _prefix_fd(49)
def i16x8_gt_u() -> bytes: return _prefix_fd(50)
def i16x8_le_s() -> bytes: return _prefix_fd(51)
def i16x8_le_u() -> bytes: return _prefix_fd(52)
def i16x8_ge_s() -> bytes: return _prefix_fd(53)
def i16x8_ge_u() -> bytes: return _prefix_fd(54)

def i32x4_eq() -> bytes: return _prefix_fd(55)
def i32x4_ne() -> bytes: return _prefix_fd(56)
def i32x4_lt_s() -> bytes: return _prefix_fd(57)
def i32x4_lt_u() -> bytes: return _prefix_fd(58)
def i32x4_gt_s() -> bytes: return _prefix_fd(59)
def i32x4_gt_u() -> bytes: return _prefix_fd(60)
def i32x4_le_s() -> bytes: return _prefix_fd(61)
def i32x4_le_u() -> bytes: return _prefix_fd(62)
def i32x4_ge_s() -> bytes: return _prefix_fd(63)
def i32x4_ge_u() -> bytes: return _prefix_fd(64)

def f32x4_eq() -> bytes: return _prefix_fd(65)
def f32x4_ne() -> bytes: return _prefix_fd(66)
def f32x4_lt() -> bytes: return _prefix_fd(67)
def f32x4_gt() -> bytes: return _prefix_fd(68)
def f32x4_le() -> bytes: return _prefix_fd(69)
def f32x4_ge() -> bytes: return _prefix_fd(70)

def f64x2_eq() -> bytes: return _prefix_fd(71)
def f64x2_ne() -> bytes: return _prefix_fd(72)
def f64x2_lt() -> bytes: return _prefix_fd(73)
def f64x2_gt() -> bytes: return _prefix_fd(74)
def f64x2_le() -> bytes: return _prefix_fd(75)
def f64x2_ge() -> bytes: return _prefix_fd(76)

# Vector bitwise
def v128_not() -> bytes: return _prefix_fd(77)
def v128_and() -> bytes: return _prefix_fd(78)
def v128_andnot() -> bytes: return _prefix_fd(79)
def v128_or() -> bytes: return _prefix_fd(80)
def v128_xor() -> bytes: return _prefix_fd(81)
def v128_bitselect() -> bytes: return _prefix_fd(82)
def v128_any_true() -> bytes: return _prefix_fd(83)

# Vector unary
def i8x16_abs() -> bytes: return _prefix_fd(96)
def i8x16_neg() -> bytes: return _prefix_fd(97)
def i8x16_popcnt() -> bytes: return _prefix_fd(98)
def i8x16_all_true() -> bytes: return _prefix_fd(99)
def i8x16_bitmask() -> bytes: return _prefix_fd(100)
def i16x8_abs() -> bytes: return _prefix_fd(128)
def i16x8_neg() -> bytes: return _prefix_fd(129)
def i16x8_all_true() -> bytes: return _prefix_fd(131)
def i16x8_bitmask() -> bytes: return _prefix_fd(132)
def i32x4_abs() -> bytes: return _prefix_fd(160)
def i32x4_neg() -> bytes: return _prefix_fd(161)
def i32x4_all_true() -> bytes: return _prefix_fd(163)
def i32x4_bitmask() -> bytes: return _prefix_fd(164)
def i64x2_abs() -> bytes: return _prefix_fd(192)
def i64x2_neg() -> bytes: return _prefix_fd(193)
def i64x2_all_true() -> bytes: return _prefix_fd(195)
def i64x2_bitmask() -> bytes: return _prefix_fd(196)

# Vector binary arithmetic
def i8x16_add() -> bytes: return _prefix_fd(110)
def i8x16_add_sat_s() -> bytes: return _prefix_fd(111)
def i8x16_add_sat_u() -> bytes: return _prefix_fd(112)
def i8x16_sub() -> bytes: return _prefix_fd(113)
def i8x16_sub_sat_s() -> bytes: return _prefix_fd(114)
def i8x16_sub_sat_u() -> bytes: return _prefix_fd(115)
def i8x16_min_s() -> bytes: return _prefix_fd(118)
def i8x16_min_u() -> bytes: return _prefix_fd(119)
def i8x16_max_s() -> bytes: return _prefix_fd(120)
def i8x16_max_u() -> bytes: return _prefix_fd(121)
def i8x16_avgr_u() -> bytes: return _prefix_fd(123)

def i16x8_add() -> bytes: return _prefix_fd(142)
def i16x8_add_sat_s() -> bytes: return _prefix_fd(143)
def i16x8_add_sat_u() -> bytes: return _prefix_fd(144)
def i16x8_sub() -> bytes: return _prefix_fd(145)
def i16x8_sub_sat_s() -> bytes: return _prefix_fd(146)
def i16x8_sub_sat_u() -> bytes: return _prefix_fd(147)
def i16x8_mul() -> bytes: return _prefix_fd(149)
def i16x8_min_s() -> bytes: return _prefix_fd(150)
def i16x8_min_u() -> bytes: return _prefix_fd(151)
def i16x8_max_s() -> bytes: return _prefix_fd(152)
def i16x8_max_u() -> bytes: return _prefix_fd(153)
def i16x8_avgr_u() -> bytes: return _prefix_fd(155)

def i32x4_add() -> bytes: return _prefix_fd(174)
def i32x4_sub() -> bytes: return _prefix_fd(177)
def i32x4_mul() -> bytes: return _prefix_fd(181)
def i32x4_min_s() -> bytes: return _prefix_fd(182)
def i32x4_min_u() -> bytes: return _prefix_fd(183)
def i32x4_max_s() -> bytes: return _prefix_fd(184)
def i32x4_max_u() -> bytes: return _prefix_fd(185)

def i64x2_add() -> bytes: return _prefix_fd(206)
def i64x2_sub() -> bytes: return _prefix_fd(209)
def i64x2_mul() -> bytes: return _prefix_fd(213)

def f32x4_add() -> bytes: return _prefix_fd(228)
def f32x4_sub() -> bytes: return _prefix_fd(229)
def f32x4_mul() -> bytes: return _prefix_fd(230)
def f32x4_div() -> bytes: return _prefix_fd(231)
def f32x4_min() -> bytes: return _prefix_fd(232)
def f32x4_max() -> bytes: return _prefix_fd(233)
def f32x4_pmin() -> bytes: return _prefix_fd(234)
def f32x4_pmax() -> bytes: return _prefix_fd(235)

def f64x2_add() -> bytes: return _prefix_fd(240)
def f64x2_sub() -> bytes: return _prefix_fd(241)
def f64x2_mul() -> bytes: return _prefix_fd(242)
def f64x2_div() -> bytes: return _prefix_fd(243)
def f64x2_min() -> bytes: return _prefix_fd(244)
def f64x2_max() -> bytes: return _prefix_fd(245)
def f64x2_pmin() -> bytes: return _prefix_fd(246)
def f64x2_pmax() -> bytes: return _prefix_fd(247)

# Vector float unary
def f32x4_abs() -> bytes: return _prefix_fd(224)
def f32x4_neg() -> bytes: return _prefix_fd(225)
def f32x4_sqrt() -> bytes: return _prefix_fd(227)
def f32x4_ceil() -> bytes: return _prefix_fd(103)
def f32x4_floor() -> bytes: return _prefix_fd(104)
def f32x4_trunc() -> bytes: return _prefix_fd(105)
def f32x4_nearest() -> bytes: return _prefix_fd(106)

def f64x2_abs() -> bytes: return _prefix_fd(236)
def f64x2_neg() -> bytes: return _prefix_fd(237)
def f64x2_sqrt() -> bytes: return _prefix_fd(239)
def f64x2_ceil() -> bytes: return _prefix_fd(116)
def f64x2_floor() -> bytes: return _prefix_fd(117)
def f64x2_trunc() -> bytes: return _prefix_fd(122)
def f64x2_nearest() -> bytes: return _prefix_fd(148)

# Vector shifts
def i8x16_shl() -> bytes: return _prefix_fd(107)
def i8x16_shr_s() -> bytes: return _prefix_fd(108)
def i8x16_shr_u() -> bytes: return _prefix_fd(109)
def i16x8_shl() -> bytes: return _prefix_fd(139)
def i16x8_shr_s() -> bytes: return _prefix_fd(140)
def i16x8_shr_u() -> bytes: return _prefix_fd(141)
def i32x4_shl() -> bytes: return _prefix_fd(171)
def i32x4_shr_s() -> bytes: return _prefix_fd(172)
def i32x4_shr_u() -> bytes: return _prefix_fd(173)
def i64x2_shl() -> bytes: return _prefix_fd(203)
def i64x2_shr_s() -> bytes: return _prefix_fd(204)
def i64x2_shr_u() -> bytes: return _prefix_fd(205)

# Vector narrow / extend
def i8x16_narrow_i16x8_s() -> bytes: return _prefix_fd(101)
def i8x16_narrow_i16x8_u() -> bytes: return _prefix_fd(102)
def i16x8_narrow_i32x4_s() -> bytes: return _prefix_fd(133)
def i16x8_narrow_i32x4_u() -> bytes: return _prefix_fd(134)

def i16x8_extend_low_i8x16_s() -> bytes: return _prefix_fd(135)
def i16x8_extend_high_i8x16_s() -> bytes: return _prefix_fd(136)
def i16x8_extend_low_i8x16_u() -> bytes: return _prefix_fd(137)
def i16x8_extend_high_i8x16_u() -> bytes: return _prefix_fd(138)
def i32x4_extend_low_i16x8_s() -> bytes: return _prefix_fd(167)
def i32x4_extend_high_i16x8_s() -> bytes: return _prefix_fd(168)
def i32x4_extend_low_i16x8_u() -> bytes: return _prefix_fd(169)
def i32x4_extend_high_i16x8_u() -> bytes: return _prefix_fd(170)
def i64x2_extend_low_i32x4_s() -> bytes: return _prefix_fd(199)
def i64x2_extend_high_i32x4_s() -> bytes: return _prefix_fd(200)
def i64x2_extend_low_i32x4_u() -> bytes: return _prefix_fd(201)
def i64x2_extend_high_i32x4_u() -> bytes: return _prefix_fd(202)

# Vector conversions
def f32x4_demote_f64x2_zero() -> bytes: return _prefix_fd(94)
def f64x2_promote_low_f32x4() -> bytes: return _prefix_fd(95)
def i32x4_trunc_sat_f32x4_s() -> bytes: return _prefix_fd(248)
def i32x4_trunc_sat_f32x4_u() -> bytes: return _prefix_fd(249)
def f32x4_convert_i32x4_s() -> bytes: return _prefix_fd(250)
def f32x4_convert_i32x4_u() -> bytes: return _prefix_fd(251)
def i32x4_trunc_sat_f64x2_s_zero() -> bytes: return _prefix_fd(252)
def i32x4_trunc_sat_f64x2_u_zero() -> bytes: return _prefix_fd(253)
def f64x2_convert_low_i32x4_s() -> bytes: return _prefix_fd(254)
def f64x2_convert_low_i32x4_u() -> bytes: return _prefix_fd(255)

# Vector extmul / dot
def i16x8_extmul_low_i8x16_s() -> bytes: return _prefix_fd(156)
def i16x8_extmul_high_i8x16_s() -> bytes: return _prefix_fd(157)
def i16x8_extmul_low_i8x16_u() -> bytes: return _prefix_fd(158)
def i16x8_extmul_high_i8x16_u() -> bytes: return _prefix_fd(159)
def i32x4_extmul_low_i16x8_s() -> bytes: return _prefix_fd(188)
def i32x4_extmul_high_i16x8_s() -> bytes: return _prefix_fd(189)
def i32x4_extmul_low_i16x8_u() -> bytes: return _prefix_fd(190)
def i32x4_extmul_high_i16x8_u() -> bytes: return _prefix_fd(191)
def i64x2_extmul_low_i32x4_s() -> bytes: return _prefix_fd(220)
def i64x2_extmul_high_i32x4_s() -> bytes: return _prefix_fd(221)
def i64x2_extmul_low_i32x4_u() -> bytes: return _prefix_fd(222)
def i64x2_extmul_high_i32x4_u() -> bytes: return _prefix_fd(223)
def i32x4_dot_i16x8_s() -> bytes: return _prefix_fd(186)
def i16x8_extadd_pairwise_i8x16_s() -> bytes: return _prefix_fd(124)
def i16x8_extadd_pairwise_i8x16_u() -> bytes: return _prefix_fd(125)
def i32x4_extadd_pairwise_i16x8_s() -> bytes: return _prefix_fd(126)
def i32x4_extadd_pairwise_i16x8_u() -> bytes: return _prefix_fd(127)

# Vector load/store lane
def v128_load8_lane(lane, offset=0, align=0) -> bytes:
    return _prefix_fd(84, _memarg(align, offset), byte(lane))
def v128_load16_lane(lane, offset=0, align=1) -> bytes:
    return _prefix_fd(85, _memarg(align, offset), byte(lane))
def v128_load32_lane(lane, offset=0, align=2) -> bytes:
    return _prefix_fd(86, _memarg(align, offset), byte(lane))
def v128_load64_lane(lane, offset=0, align=3) -> bytes:
    return _prefix_fd(87, _memarg(align, offset), byte(lane))
def v128_store8_lane(lane, offset=0, align=0) -> bytes:
    return _prefix_fd(88, _memarg(align, offset), byte(lane))
def v128_store16_lane(lane, offset=0, align=1) -> bytes:
    return _prefix_fd(89, _memarg(align, offset), byte(lane))
def v128_store32_lane(lane, offset=0, align=2) -> bytes:
    return _prefix_fd(90, _memarg(align, offset), byte(lane))
def v128_store64_lane(lane, offset=0, align=3) -> bytes:
    return _prefix_fd(91, _memarg(align, offset), byte(lane))

# Vector load extend/splat
def v128_load8x8_s(offset=0, align=1) -> bytes: return _prefix_fd(1, _memarg(align, offset))
def v128_load8x8_u(offset=0, align=1) -> bytes: return _prefix_fd(2, _memarg(align, offset))
def v128_load16x4_s(offset=0, align=2) -> bytes: return _prefix_fd(3, _memarg(align, offset))
def v128_load16x4_u(offset=0, align=2) -> bytes: return _prefix_fd(4, _memarg(align, offset))
def v128_load32x2_s(offset=0, align=3) -> bytes: return _prefix_fd(5, _memarg(align, offset))
def v128_load32x2_u(offset=0, align=3) -> bytes: return _prefix_fd(6, _memarg(align, offset))
def v128_load8_splat(offset=0, align=0) -> bytes: return _prefix_fd(7, _memarg(align, offset))
def v128_load16_splat(offset=0, align=1) -> bytes: return _prefix_fd(8, _memarg(align, offset))
def v128_load32_splat(offset=0, align=2) -> bytes: return _prefix_fd(9, _memarg(align, offset))
def v128_load64_splat(offset=0, align=3) -> bytes: return _prefix_fd(10, _memarg(align, offset))
def v128_load32_zero(offset=0, align=2) -> bytes: return _prefix_fd(92, _memarg(align, offset))
def v128_load64_zero(offset=0, align=3) -> bytes: return _prefix_fd(93, _memarg(align, offset))

# Relaxed SIMD
def i8x16_relaxed_swizzle() -> bytes: return _prefix_fd(256)
def i32x4_relaxed_trunc_f32x4_s() -> bytes: return _prefix_fd(257)
def i32x4_relaxed_trunc_f32x4_u() -> bytes: return _prefix_fd(258)
def i32x4_relaxed_trunc_f64x2_s_zero() -> bytes: return _prefix_fd(259)
def i32x4_relaxed_trunc_f64x2_u_zero() -> bytes: return _prefix_fd(260)
def f32x4_relaxed_madd() -> bytes: return _prefix_fd(261)
def f32x4_relaxed_nmadd() -> bytes: return _prefix_fd(262)
def f64x2_relaxed_madd() -> bytes: return _prefix_fd(263)
def f64x2_relaxed_nmadd() -> bytes: return _prefix_fd(264)
def i8x16_relaxed_laneselect() -> bytes: return _prefix_fd(265)
def i16x8_relaxed_laneselect() -> bytes: return _prefix_fd(266)
def i32x4_relaxed_laneselect() -> bytes: return _prefix_fd(267)
def i64x2_relaxed_laneselect() -> bytes: return _prefix_fd(268)
def f32x4_relaxed_min() -> bytes: return _prefix_fd(269)
def f32x4_relaxed_max() -> bytes: return _prefix_fd(270)
def f64x2_relaxed_min() -> bytes: return _prefix_fd(271)
def f64x2_relaxed_max() -> bytes: return _prefix_fd(272)
def i16x8_relaxed_q15mulr_s() -> bytes: return _prefix_fd(273)
def i16x8_relaxed_dot() -> bytes: return _prefix_fd(274)
def i32x4_relaxed_dot_add() -> bytes: return _prefix_fd(275)

# i64x2 comparisons (extended)
def i64x2_eq() -> bytes: return _prefix_fd(214)
def i64x2_ne() -> bytes: return _prefix_fd(215)
def i64x2_lt_s() -> bytes: return _prefix_fd(216)
def i64x2_gt_s() -> bytes: return _prefix_fd(217)
def i64x2_le_s() -> bytes: return _prefix_fd(218)
def i64x2_ge_s() -> bytes: return _prefix_fd(219)

# i16x8 q15mulr
def i16x8_q15mulr_sat_s() -> bytes: return _prefix_fd(130)


# -- Expression --

def expr(*instrs) -> bytes:
    """Encode an expression: instructions + end byte (0x0B)."""
    return b''.join(instrs) + byte(0x0B)


# ===================================================================
# Section encoders
# ===================================================================

def custom_section(name_str: str, data: bytes) -> bytes:
    """Encode a custom section."""
    payload = name(name_str) + data
    return section(SEC_CUSTOM, payload)

def type_section(types) -> bytes:
    """Encode the type section. types is a list of encoded type definitions."""
    return section(SEC_TYPE, vec(types, raw))

def import_section(imports) -> bytes:
    """Encode the import section. imports is a list of encoded import entries."""
    return section(SEC_IMPORT, vec(imports, raw))

def func_section(type_indices) -> bytes:
    """Encode the function section. type_indices is a list of type indices."""
    return section(SEC_FUNC, vec(type_indices, u32))

def table_section(tables) -> bytes:
    """Encode the table section. tables is a list of encoded table entries."""
    return section(SEC_TABLE, vec(tables, raw))

def memory_section(memories) -> bytes:
    """Encode the memory section. memories is a list of encoded memory entries."""
    return section(SEC_MEMORY, vec(memories, raw))

def tag_section(tags) -> bytes:
    """Encode the tag section. tags is a list of encoded tag entries."""
    return section(SEC_TAG, vec(tags, raw))

def global_section(globals_) -> bytes:
    """Encode the global section. globals_ is a list of encoded global entries."""
    return section(SEC_GLOBAL, vec(globals_, raw))

def export_section(exports) -> bytes:
    """Encode the export section. exports is a list of encoded export entries."""
    return section(SEC_EXPORT, vec(exports, raw))

def start_section(funcidx: int) -> bytes:
    """Encode the start section."""
    return section(SEC_START, u32(funcidx))

def element_section(elements) -> bytes:
    """Encode the element section. elements is a list of encoded element segments."""
    return section(SEC_ELEM, vec(elements, raw))

def data_count_section(count: int) -> bytes:
    """Encode the data count section."""
    return section(SEC_DATACOUNT, u32(count))

def code_section(codes) -> bytes:
    """Encode the code section. codes is a list of encoded code entries."""
    return section(SEC_CODE, vec(codes, raw))

def data_section(datas) -> bytes:
    """Encode the data section. datas is a list of encoded data segments."""
    return section(SEC_DATA, vec(datas, raw))


# ===================================================================
# Convenience builders for section entries
# ===================================================================

# -- Import entries --

def import_func(module_name: str, field_name: str, typeidx: int) -> bytes:
    """Encode a function import."""
    return name(module_name) + name(field_name) + externtype_func(typeidx)

def import_table(module_name: str, field_name: str,
                 ref_null: bool, ht, min_val: int, max_val=None) -> bytes:
    """Encode a table import."""
    return name(module_name) + name(field_name) + \
           externtype_table(ref_null, ht, min_val, max_val)

def import_memory(module_name: str, field_name: str,
                  min_val: int, max_val=None) -> bytes:
    """Encode a memory import."""
    return name(module_name) + name(field_name) + \
           externtype_mem(min_val, max_val)

def import_global(module_name: str, field_name: str,
                  vt, mutable: bool = False) -> bytes:
    """Encode a global import."""
    return name(module_name) + name(field_name) + \
           externtype_global(vt, mutable)

def import_tag(module_name: str, field_name: str, typeidx: int) -> bytes:
    """Encode a tag import."""
    return name(module_name) + name(field_name) + \
           externtype_tag(typeidx)


# -- Export entries --

def export_func(name_str: str, idx: int) -> bytes:
    """Encode a function export."""
    return name(name_str) + byte(EXT_FUNC) + u32(idx)

def export_table(name_str: str, idx: int) -> bytes:
    """Encode a table export."""
    return name(name_str) + byte(EXT_TABLE) + u32(idx)

def export_memory(name_str: str, idx: int) -> bytes:
    """Encode a memory export."""
    return name(name_str) + byte(EXT_MEM) + u32(idx)

def export_global(name_str: str, idx: int) -> bytes:
    """Encode a global export."""
    return name(name_str) + byte(EXT_GLOBAL) + u32(idx)

def export_tag(name_str: str, idx: int) -> bytes:
    """Encode a tag export."""
    return name(name_str) + byte(EXT_TAG) + u32(idx)


# -- Table entries --

def table_entry(ref_null: bool, ht, min_val: int, max_val=None) -> bytes:
    """Encode a table entry (simple form: type + limits)."""
    return tabletype(ref_null, ht, min_val, max_val)

def table_entry_with_init(ref_null, ht, min_val, max_val, init_expr) -> bytes:
    """Encode a table entry with explicit initializer expression."""
    return byte(0x40) + byte(0x00) + \
           tabletype(ref_null, ht, min_val, max_val) + \
           expr(*init_expr)


# -- Memory entries --

def memory_entry(min_val: int, max_val=None) -> bytes:
    """Encode a memory entry."""
    return memtype(min_val, max_val)


# -- Global entries --

def global_entry(vt, mutable: bool, init_instrs) -> bytes:
    """Encode a global entry: type + init expression.

    init_instrs: list of instruction bytes for the init expression.
    """
    return globaltype(vt, mutable) + expr(*init_instrs)


# -- Tag entries --

def tag_entry(typeidx: int) -> bytes:
    """Encode a tag entry."""
    return tagtype(typeidx)


# -- Element segments --

def elem_active(offset_instrs, func_indices) -> bytes:
    """Active element on table 0 with function indices (variant 0)."""
    return u32(0) + expr(*offset_instrs) + vec(func_indices, u32)

def elem_active_table(tableidx: int, offset_instrs, func_indices) -> bytes:
    """Active element on specific table with function indices (variant 2)."""
    return u32(2) + u32(tableidx) + expr(*offset_instrs) + \
           byte(0x00) + vec(func_indices, u32)

def elem_passive(func_indices) -> bytes:
    """Passive element with function indices (variant 1)."""
    return u32(1) + byte(0x00) + vec(func_indices, u32)

def elem_declare(func_indices) -> bytes:
    """Declarative element with function indices (variant 3)."""
    return u32(3) + byte(0x00) + vec(func_indices, u32)

def elem_active_expr(offset_instrs, init_exprs) -> bytes:
    """Active element on table 0 with arbitrary init expressions (variant 4)."""
    return u32(4) + expr(*offset_instrs) + vec(init_exprs, lambda e: expr(*e))

def elem_passive_expr(ref_null, ht, init_exprs) -> bytes:
    """Passive element with arbitrary init expressions (variant 5)."""
    return u32(5) + reftype(ref_null, ht) + vec(init_exprs, lambda e: expr(*e))

def elem_active_table_expr(tableidx, offset_instrs, ref_null, ht, init_exprs) -> bytes:
    """Active element on specific table with arbitrary init expressions (variant 6)."""
    return u32(6) + u32(tableidx) + expr(*offset_instrs) + \
           reftype(ref_null, ht) + vec(init_exprs, lambda e: expr(*e))

def elem_declare_expr(ref_null, ht, init_exprs) -> bytes:
    """Declarative element with arbitrary init expressions (variant 7)."""
    return u32(7) + reftype(ref_null, ht) + vec(init_exprs, lambda e: expr(*e))


# -- Data segments --

def data_active(offset_instrs, data_bytes) -> bytes:
    """Active data on memory 0 (variant 0)."""
    return u32(0) + expr(*offset_instrs) + vec(data_bytes, byte)

def data_passive(data_bytes) -> bytes:
    """Passive data (variant 1)."""
    return u32(1) + vec(data_bytes, byte)

def data_active_mem(memidx: int, offset_instrs, data_bytes) -> bytes:
    """Active data on specific memory (variant 2)."""
    return u32(2) + u32(memidx) + expr(*offset_instrs) + vec(data_bytes, byte)


# -- Code entries --

def func_body(locals_, instrs) -> bytes:
    """Encode a function body (code entry).

    locals_: list of (count, valtype) tuples for local declarations.
    valtype may be either an int (single-byte numeric type) or raw
    bytes (for reftypes, which are multi-byte).
    instrs: list of instruction bytes

    The body is size-prefixed per the Bcode grammar.
    """
    def _enc_local(l):
        count, vt = l
        vt_bytes = vt if isinstance(vt, (bytes, bytearray)) else byte(vt)
        return u32(count) + vt_bytes
    encoded_locals = vec(locals_, _enc_local)
    encoded_expr = b''.join(instrs) + byte(0x0B)
    body = encoded_locals + encoded_expr
    return u32(len(body)) + body


# ===================================================================
# Module encoder
# ===================================================================

def module(*,
           types=None,
           imports=None,
           funcs=None,
           tables=None,
           memories=None,
           tags=None,
           globals_=None,
           exports=None,
           start=None,
           elements=None,
           data_count=None,
           codes=None,
           datas=None,
           customs=None):
    """Encode a complete WebAssembly 3.0 module.

    All parameters are optional. Sections are emitted in the correct order.
    Customs can be a list of encoded custom sections to intersperse.

    Returns the complete .wasm binary as bytes.

    Example::

        wasm = module(
            types=[functype([I32, I32], [I32])],
            funcs=[0],  # type index
            codes=[func_body([], [local_get(0), local_get(1), i32_add()])],
            exports=[export_func("add", 0)],
        )
    """
    parts = [WASM_MAGIC, WASM_VERSION]

    if customs:
        for c in customs:
            parts.append(c)

    if types is not None:
        parts.append(type_section(types))

    if imports is not None:
        parts.append(import_section(imports))

    if funcs is not None:
        parts.append(func_section(funcs))

    if tables is not None:
        parts.append(table_section(tables))

    if memories is not None:
        parts.append(memory_section(memories))

    if tags is not None:
        parts.append(tag_section(tags))

    if globals_ is not None:
        parts.append(global_section(globals_))

    if exports is not None:
        parts.append(export_section(exports))

    if start is not None:
        parts.append(start_section(start))

    if elements is not None:
        parts.append(element_section(elements))

    if data_count is not None:
        parts.append(data_count_section(data_count))

    if codes is not None:
        parts.append(code_section(codes))

    if datas is not None:
        parts.append(data_section(datas))

    return b''.join(parts)
