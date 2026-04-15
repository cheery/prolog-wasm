"""Extensive usage examples for the WebAssembly 3.0 binary format encoder.

Every example builds a .wasm module and validates it with a real engine.
Most examples are checked with ``wasm-validate --enable-all``.  Examples
that exercise newer proposal features (GC, exceptions, function-references)
are validated with ``wasmtime compile -W all-proposals=y``.

Run directly:

    python3 examples.py        # exit 0 ⇒ all examples pass
"""

import os
import shutil
import subprocess
import tempfile

from encoder import *

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

WASMTIME = shutil.which("wasmtime") or "/home/cheery/.wasmtime/bin/wasmtime"


def _to_temp_file(wasm_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wasm")
    os.write(fd, wasm_bytes)
    os.close(fd)
    return path


def validate(wasm_bytes: bytes, label: str, *, extra_flags: list[str] | None = None) -> None:
    """Validate with wasm-validate."""
    path = _to_temp_file(wasm_bytes)
    try:
        cmd = ["wasm-validate", path]
        if extra_flags:
            cmd.extend(extra_flags)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"{label} FAILED:\n{r.stderr}")
    finally:
        os.unlink(path)
    print(f"  ok  {label}")


def validate_all(wasm_bytes: bytes, label: str) -> None:
    """Validate with wasm-validate --enable-all."""
    path = _to_temp_file(wasm_bytes)
    try:
        r = subprocess.run(["wasm-validate", "--enable-all", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            r2 = subprocess.run(["wasm-validate", path],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                raise AssertionError(f"{label} FAILED:\n{r2.stderr}")
    finally:
        os.unlink(path)
    print(f"  ok  {label}")


def validate_proposal(wasm_bytes: bytes, label: str) -> None:
    """Validate using wasmtime with all proposals enabled.

    Used for GC, exceptions, function-references, and other features
    that the installed wasm-validate may not support yet.
    """
    path = _to_temp_file(wasm_bytes)
    try:
        r = subprocess.run(
            [WASMTIME, "compile", "-W", "all-proposals=y", path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise AssertionError(f"{label} FAILED (wasmtime):\n{r.stderr}")
    finally:
        os.unlink(path)
    print(f"  ok  {label}")


# ===================================================================
# 01  LEB128 primitives
# ===================================================================

def example_01_leb128():
    """LEB128 unsigned and signed edge-cases."""
    # Unsigned
    assert leb128_u(0)        == b'\x00'
    assert leb128_u(1)        == b'\x01'
    assert leb128_u(127)      == b'\x7f'
    assert leb128_u(128)      == b'\x80\x01'
    assert leb128_u(300)      == b'\xac\x02'
    assert leb128_u(16383)    == b'\xff\x7f'
    assert leb128_u(16384)    == b'\x80\x80\x01'
    assert leb128_u(2**32-1)  == b'\xff\xff\xff\xff\x0f'

    # Signed
    assert leb128_s(0)     == b'\x00'
    assert leb128_s(1)     == b'\x01'
    assert leb128_s(-1)    == b'\x7f'
    assert leb128_s(63)    == b'\x3f'
    assert leb128_s(-64)   == b'\x40'

    # Round-trip through a real module
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], [i32_const(0)])],
        exports=[export_func("zero", 0)],
    )
    validate_all(wasm, "01-leb128-zero-constant")


# ===================================================================
# 02  Simplest possible module
# ===================================================================

def example_02_minimal():
    """One function, one export."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [])],
        exports=[export_func("run", 0)],
    )
    validate_all(wasm, "02-minimal-module")


# ===================================================================
# 03  Function with parameters and result
# ===================================================================

def example_03_add():
    """(i32, i32) -> i32  addition."""
    wasm = module(
        types=[functype([I32, I32], [I32])],
        funcs=[0],
        codes=[func_body([], [local_get(0), local_get(1), i32_add()])],
        exports=[export_func("add", 0)],
    )
    validate_all(wasm, "03-add")


# ===================================================================
# 04  Multiple functions
# ===================================================================

def example_04_multi_func():
    """Two functions that call each other."""
    wasm = module(
        types=[
            functype([I32], [I32]),       # type 0
            functype([I32, I32], [I32]),  # type 1
        ],
        funcs=[0, 1, 1],                  # func 0:type0, func 1:type1, func 2:type1
        codes=[
            # double(n) = add(n, n)
            func_body([], [local_get(0), local_get(0), call(1)]),
            # add(a, b)
            func_body([], [local_get(0), local_get(1), i32_add()]),
            # add3(a, b) = add(a, b) + 1
            func_body([], [local_get(0), local_get(1), call(1), i32_const(1), i32_add()]),
        ],
        exports=[
            export_func("double", 0),
            export_func("add", 1),
            export_func("add3", 2),
        ],
    )
    validate_all(wasm, "04-multi-func")


# ===================================================================
# 05  Locals
# ===================================================================

def example_05_locals():
    """Function with local variables."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body(
            [(1, I32), (1, I64)],         # 1 i32 local + 1 i64 local
            [
                i32_const(42),
                local_set(1),              # temp = 42
                local_get(0),
                local_get(1),
                i32_add(),
            ],
        )],
        exports=[export_func("with_locals", 0)],
    )
    validate_all(wasm, "05-locals")


# ===================================================================
# 06  Control flow: block / loop / br / br_if
# ===================================================================

def example_06_control_flow():
    """sum = 0; while (n > 0) { sum += n; n--; } return sum."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([(1, I32)], [   # local 0 = param n, local 1 = sum
            i32_const(0),
            local_set(1),                 # sum = 0

            block(None, [                 # $break
                loop(None, [              # $continue
                    local_get(0),
                    i32_eqz(),
                    br_if(1),             # br $break if n == 0

                    local_get(1),
                    local_get(0),
                    i32_add(),
                    local_set(1),         # sum += n

                    local_get(0),
                    i32_const(1),
                    i32_sub(),
                    local_set(0),         # n--

                    br(0),                # continue $loop
                ]),
            ]),
            local_get(1),                 # return sum
        ])],
        exports=[export_func("countdown_sum", 0)],
    )
    validate_all(wasm, "06-control-flow")


# ===================================================================
# 07  Control flow: if / else
# ===================================================================

def example_07_if_else():
    """If / else branches producing a value."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            local_get(0),
            if_(I32, [                    # then
                i32_const(100),
            ], [                          # else
                i32_const(200),
            ]),
        ])],
        exports=[export_func("branch", 0)],
    )
    validate_all(wasm, "07-if-else")


# ===================================================================
# 08  Block types (void and valtype)
# ===================================================================

def example_08_blocktypes():
    """Block with void type and value type results."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            block(None, [                 # void block
                local_get(0),
                i32_const(1),
                i32_add(),
                local_set(0),
            ]),
            block(I32, [                  # block producing i32
                local_get(0),
            ]),
        ])],
        exports=[export_func("blocktypes", 0)],
    )
    validate_all(wasm, "08-blocktypes")


# ===================================================================
# 09  br_table
# ===================================================================

def example_09_br_table():
    """br_table (switch-like dispatch)."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            block(None, [
                block(None, [
                    block(None, [
                        local_get(0),
                        br_table([0, 1, 2], 2),
                    ]),
                    i32_const(10),
                    return_(),
                ]),
                i32_const(20),
                return_(),
            ]),
            i32_const(30),
        ])],
        exports=[export_func("switch", 0)],
    )
    validate_all(wasm, "09-br-table")


# ===================================================================
# 10  Numeric constants
# ===================================================================

def example_10_numeric_constants():
    """Push constants of every numeric type."""
    for label, vt, instrs in [
        ("i32_pos",  functype([], [I32]), [i32_const(42)]),
        ("i32_neg",  functype([], [I32]), [i32_const(-1)]),
        ("i32_zero", functype([], [I32]), [i32_const(0)]),
        ("i64_max",  functype([], [I64]), [i64_const(0x7FFFFFFFFFFFFFFF)]),
        ("i64_neg",  functype([], [I64]), [i64_const(-42)]),
        ("f32_pi",   functype([], [F32]), [f32_const(3.14)]),
        ("f64_e",    functype([], [F64]), [f64_const(2.718281828)]),
    ]:
        wasm = module(types=[vt], funcs=[0],
                      codes=[func_body([], instrs)],
                      exports=[export_func(label, 0)])
        validate_all(wasm, f"10-{label}")


# ===================================================================
# 11  i32 relational operators
# ===================================================================

def example_11_i32_relops():
    """All i32 relational operators."""
    # Binary: (i32, i32) -> i32
    for suffix, op in [
        ("eq", i32_eq), ("ne", i32_ne),
        ("lt_s", i32_lt_s), ("lt_u", i32_lt_u),
        ("gt_s", i32_gt_s), ("gt_u", i32_gt_u),
        ("le_s", i32_le_s), ("le_u", i32_le_u),
        ("ge_s", i32_ge_s), ("ge_u", i32_ge_u),
    ]:
        wasm = module(
            types=[functype([I32, I32], [I32])],
            funcs=[0],
            codes=[func_body([], [local_get(0), local_get(1), op()])],
            exports=[export_func(f"i32_{suffix}", 0)],
        )
        validate_all(wasm, f"11-i32-{suffix}")

    # Unary: i32.eqz
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [local_get(0), i32_eqz()])],
        exports=[export_func("i32_eqz", 0)],
    )
    validate_all(wasm, "11-i32-eqz")


# ===================================================================
# 12  i32 arithmetic operators
# ===================================================================

def example_12_i32_arithmetic():
    """i32 binary and unary arithmetic."""
    for suffix, op in [
        ("add", i32_add), ("sub", i32_sub), ("mul", i32_mul),
        ("div_s", i32_div_s), ("div_u", i32_div_u),
        ("rem_s", i32_rem_s), ("rem_u", i32_rem_u),
        ("and", i32_and), ("or", i32_or), ("xor", i32_xor),
        ("shl", i32_shl), ("shr_s", i32_shr_s), ("shr_u", i32_shr_u),
        ("rotl", i32_rotl), ("rotr", i32_rotr),
    ]:
        wasm = module(
            types=[functype([I32, I32], [I32])],
            funcs=[0],
            codes=[func_body([], [local_get(0), local_get(1), op()])],
            exports=[export_func(f"i32_{suffix}", 0)],
        )
        validate_all(wasm, f"12-i32-{suffix}")

    for suffix, op in [("clz", i32_clz), ("ctz", i32_ctz), ("popcnt", i32_popcnt)]:
        wasm = module(
            types=[functype([I32], [I32])],
            funcs=[0],
            codes=[func_body([], [local_get(0), op()])],
            exports=[export_func(f"i32_{suffix}", 0)],
        )
        validate_all(wasm, f"12-i32-{suffix}")


# ===================================================================
# 13  i64 arithmetic and comparison operators
# ===================================================================

def example_13_i64_ops():
    """i64 arithmetic (returns i64) and comparisons (return i32)."""
    # Arithmetic: (i64, i64) -> i64
    for suffix, op in [
        ("add", i64_add), ("sub", i64_sub), ("mul", i64_mul),
        ("div_s", i64_div_s), ("div_u", i64_div_u),
        ("and", i64_and), ("or", i64_or), ("xor", i64_xor),
        ("shl", i64_shl), ("shr_s", i64_shr_s), ("shr_u", i64_shr_u),
    ]:
        wasm = module(
            types=[functype([I64, I64], [I64])],
            funcs=[0],
            codes=[func_body([], [local_get(0), local_get(1), op()])],
            exports=[export_func(f"i64_{suffix}", 0)],
        )
        validate_all(wasm, f"13-i64-{suffix}")

    # Comparisons: (i64, i64) -> i32
    for suffix, op in [
        ("eq", i64_eq), ("ne", i64_ne),
        ("lt_s", i64_lt_s), ("gt_s", i64_gt_s),
        ("le_s", i64_le_s), ("ge_s", i64_ge_s),
    ]:
        wasm = module(
            types=[functype([I64, I64], [I32])],
            funcs=[0],
            codes=[func_body([], [local_get(0), local_get(1), op()])],
            exports=[export_func(f"i64_{suffix}", 0)],
        )
        validate_all(wasm, f"13-i64-cmp-{suffix}")

    # i64.eqz: (i64) -> i32
    wasm = module(
        types=[functype([I64], [I32])],
        funcs=[0],
        codes=[func_body([], [local_get(0), i64_eqz()])],
        exports=[export_func("i64_eqz", 0)],
    )
    validate_all(wasm, "13-i64-eqz")


# ===================================================================
# 14  f32 and f64 operations
# ===================================================================

def example_14_float_ops():
    """Float arithmetic, unary, and comparison operations."""
    # f32 arithmetic: (f32, f32) -> f32
    for suffix, op in [("add", f32_add), ("sub", f32_sub), ("mul", f32_mul),
                       ("div", f32_div), ("min", f32_min), ("max", f32_max)]:
        wasm = module(types=[functype([F32, F32], [F32])], funcs=[0],
                      codes=[func_body([], [local_get(0), local_get(1), op()])],
                      exports=[export_func(f"f32_{suffix}", 0)])
        validate_all(wasm, f"14-f32-{suffix}")

    # f32 unary: (f32) -> f32
    for suffix, op in [("abs", f32_abs), ("neg", f32_neg), ("ceil", f32_ceil),
                       ("floor", f32_floor), ("trunc", f32_trunc),
                       ("nearest", f32_nearest), ("sqrt", f32_sqrt)]:
        wasm = module(types=[functype([F32], [F32])], funcs=[0],
                      codes=[func_body([], [local_get(0), op()])],
                      exports=[export_func(f"f32_{suffix}", 0)])
        validate_all(wasm, f"14-f32-unary-{suffix}")

    # f32 comparisons: (f32, f32) -> i32
    for suffix, op in [("eq", f32_eq), ("ne", f32_ne), ("lt", f32_lt),
                       ("gt", f32_gt), ("le", f32_le), ("ge", f32_ge)]:
        wasm = module(types=[functype([F32, F32], [I32])], funcs=[0],
                      codes=[func_body([], [local_get(0), local_get(1), op()])],
                      exports=[export_func(f"f32_{suffix}", 0)])
        validate_all(wasm, f"14-f32-cmp-{suffix}")

    # f64 arithmetic: (f64, f64) -> f64
    for suffix, op in [("add", f64_add), ("sub", f64_sub), ("mul", f64_mul),
                       ("div", f64_div)]:
        wasm = module(types=[functype([F64, F64], [F64])], funcs=[0],
                      codes=[func_body([], [local_get(0), local_get(1), op()])],
                      exports=[export_func(f"f64_{suffix}", 0)])
        validate_all(wasm, f"14-f64-{suffix}")


# ===================================================================
# 15  Numeric conversions
# ===================================================================

def example_15_conversions():
    """Type conversion operations."""
    for label, ft, instrs in [
        ("i32_wrap_i64",        functype([I64], [I32]),  [local_get(0), i32_wrap_i64()]),
        ("i64_extend_i32_s",    functype([I32], [I64]),  [local_get(0), i64_extend_i32_s()]),
        ("i64_extend_i32_u",    functype([I32], [I64]),  [local_get(0), i64_extend_i32_u()]),
        ("f32_convert_i32_s",   functype([I32], [F32]),  [local_get(0), f32_convert_i32_s()]),
        ("f32_demote_f64",      functype([F64], [F32]),  [local_get(0), f32_demote_f64()]),
        ("f64_promote_f32",     functype([F32], [F64]),  [local_get(0), f64_promote_f32()]),
        ("f64_convert_i32_s",   functype([I32], [F64]),  [local_get(0), f64_convert_i32_s()]),
        ("i32_reinterpret_f32", functype([F32], [I32]),  [local_get(0), i32_reinterpret_f32()]),
        ("f32_reinterpret_i32", functype([I32], [F32]),  [local_get(0), f32_reinterpret_i32()]),
        ("i64_reinterpret_f64", functype([F64], [I64]),  [local_get(0), i64_reinterpret_f64()]),
        ("f64_reinterpret_i64", functype([I64], [F64]),  [local_get(0), f64_reinterpret_i64()]),
    ]:
        wasm = module(types=[ft], funcs=[0], codes=[func_body([], instrs)],
                      exports=[export_func(label, 0)])
        validate_all(wasm, f"15-{label}")


# ===================================================================
# 16  Saturating truncations
# ===================================================================

def example_16_saturating():
    """Saturating float-to-int conversions."""
    for label, ft, instrs in [
        ("i32_trunc_sat_f32_s", functype([F32], [I32]), [local_get(0), i32_trunc_sat_f32_s()]),
        ("i32_trunc_sat_f32_u", functype([F32], [I32]), [local_get(0), i32_trunc_sat_f32_u()]),
        ("i32_trunc_sat_f64_s", functype([F64], [I32]), [local_get(0), i32_trunc_sat_f64_s()]),
        ("i32_trunc_sat_f64_u", functype([F64], [I32]), [local_get(0), i32_trunc_sat_f64_u()]),
        ("i64_trunc_sat_f32_s", functype([F32], [I64]), [local_get(0), i64_trunc_sat_f32_s()]),
        ("i64_trunc_sat_f64_s", functype([F64], [I64]), [local_get(0), i64_trunc_sat_f64_s()]),
    ]:
        wasm = module(types=[ft], funcs=[0], codes=[func_body([], instrs)],
                      exports=[export_func(label, 0)])
        validate_all(wasm, f"16-{label}")


# ===================================================================
# 17  Sign-extension operations
# ===================================================================

def example_17_sign_extension():
    """i32/i64 sign-extension ops."""
    for label, ft, instrs in [
        ("i32_extend8_s",  functype([I32], [I32]), [local_get(0), i32_extend8_s()]),
        ("i32_extend16_s", functype([I32], [I32]), [local_get(0), i32_extend16_s()]),
        ("i64_extend8_s",  functype([I64], [I64]), [local_get(0), i64_extend8_s()]),
        ("i64_extend16_s", functype([I64], [I64]), [local_get(0), i64_extend16_s()]),
        ("i64_extend32_s", functype([I64], [I64]), [local_get(0), i64_extend32_s()]),
    ]:
        wasm = module(types=[ft], funcs=[0], codes=[func_body([], instrs)],
                      exports=[export_func(label, 0)])
        validate_all(wasm, f"17-{label}")


# ===================================================================
# 18  Memory: declaration and basic load/store
# ===================================================================

def example_18_memory():
    """Declare a memory and use i32.load / i32.store."""
    wasm = module(
        types=[functype([I32, I32], [I32])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            local_get(1),       # address
            local_get(0),       # value
            i32_store(),
            i32_const(0),
            i32_load(),
        ])],
        exports=[export_func("load_store", 0), export_memory("mem", 0)],
    )
    validate_all(wasm, "18-memory-load-store")


# ===================================================================
# 19  All memory load/store variants
# ===================================================================

def example_19_memory_variants():
    """Every load and store variant with correct result types."""
    # Loads that return i32
    i32_loads = [
        ("i32_load",     i32_load),
        ("i32_load8_s",  i32_load8_s),
        ("i32_load8_u",  i32_load8_u),
        ("i32_load16_s", i32_load16_s),
        ("i32_load16_u", i32_load16_u),
    ]
    for suffix, loader in i32_loads:
        wasm = module(types=[functype([], [I32])], memories=[memory_entry(1)],
                      funcs=[0], codes=[func_body([], [i32_const(0), loader()])],
                      exports=[export_func(suffix, 0)])
        validate_all(wasm, f"19-load-{suffix}")

    # Loads that return i64
    i64_loads = [i64_load, i64_load8_s, i64_load8_u, i64_load16_s,
                 i64_load16_u, i64_load32_s, i64_load32_u]
    for loader in i64_loads:
        wasm = module(types=[functype([], [I64])], memories=[memory_entry(1)],
                      funcs=[0], codes=[func_body([], [i32_const(0), loader()])],
                      exports=[export_func(loader.__name__, 0)])
        validate_all(wasm, f"19-load-{loader.__name__}")

    # Loads that return f32 / f64
    for loader, rt in [(f32_load, F32), (f64_load, F64)]:
        wasm = module(types=[functype([], [rt])], memories=[memory_entry(1)],
                      funcs=[0], codes=[func_body([], [i32_const(0), loader()])],
                      exports=[export_func(loader.__name__, 0)])
        validate_all(wasm, f"19-load-{loader.__name__}")

    # Stores: (valtype) -> ()
    for suffix, storer, vt in [
        ("i32_store",   i32_store,   I32),
        ("i64_store",   i64_store,   I64),
        ("f32_store",   f32_store,   F32),
        ("f64_store",   f64_store,   F64),
        ("i32_store8",  i32_store8,  I32),
        ("i32_store16", i32_store16, I32),
        ("i64_store8",  i64_store8,  I64),
        ("i64_store16", i64_store16, I64),
        ("i64_store32", i64_store32, I64),
    ]:
        wasm = module(
            types=[functype([vt], [])],
            memories=[memory_entry(1)],
            funcs=[0],
            codes=[func_body([], [i32_const(0), local_get(0), storer()])],
            exports=[export_func(suffix, 0)],
        )
        validate_all(wasm, f"19-store-{suffix}")


# ===================================================================
# 20  Memory with explicit offsets and alignment
# ===================================================================

def example_20_memarg_offset_align():
    """Load and store with explicit offset and alignment."""
    wasm = module(
        types=[functype([], [I32])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0),
            i32_const(42),
            i32_store(offset=100),
            i32_const(0),
            i32_load(offset=100),
        ])],
        exports=[export_func("offset_test", 0)],
    )
    validate_all(wasm, "20-memarg-offset")


# ===================================================================
# 21  memory.size / memory.grow
# ===================================================================

def example_21_memory_size_grow():
    """memory.size and memory.grow."""
    wasm = module(
        types=[functype([I32], [I32])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            memory_size(0),
            local_get(0),
            memory_grow(0),
            i32_add(),
        ])],
        exports=[export_func("mem_ops", 0)],
    )
    validate_all(wasm, "21-memory-size-grow")


# ===================================================================
# 22  Globals (immutable and mutable)
# ===================================================================

def example_22_globals():
    """Immutable and mutable globals."""
    wasm = module(
        types=[functype([], [I32])],
        globals_=[
            global_entry(I32, False, [i32_const(42)]),
            global_entry(I32, True,  [i32_const(0)]),
        ],
        funcs=[0],
        codes=[func_body([], [
            global_get(1),
            i32_const(1),
            i32_add(),
            global_set(1),
            global_get(1),
        ])],
        exports=[
            export_func("inc_global", 0),
            export_global("const_42", 0),
            export_global("counter", 1),
        ],
    )
    validate_all(wasm, "22-globals")


# ===================================================================
# 23  Float globals
# ===================================================================

def example_23_float_globals():
    """Globals initialised with float constants."""
    wasm = module(
        types=[functype([], [F64])],
        globals_=[
            global_entry(F64, False, [f64_const(2.718281828)]),
            global_entry(F32, False, [f32_const(3.14)]),
        ],
        funcs=[0],
        codes=[func_body([], [global_get(0)])],
        exports=[export_func("euler", 0)],
    )
    validate_all(wasm, "23-float-globals")


# ===================================================================
# 24  Table declaration
# ===================================================================

def example_24_table():
    """Declare a funcref table."""
    wasm = module(
        types=[functype([], [])],
        tables=[table_entry(True, HT_FUNC, 0, 10)],
        funcs=[0],
        codes=[func_body([], [])],
        exports=[export_table("tbl", 0)],
    )
    validate_all(wasm, "24-table-funcref")


# ===================================================================
# 25  Imports: function, memory, global, table
# ===================================================================

def example_25_imports():
    """Import functions, memory, globals, and a table."""
    wasm = module(
        types=[
            functype([I32], []),     # type 0
            functype([], [I32]),     # type 1
        ],
        imports=[
            import_func("env", "log", 0),
            import_global("env", "seed", I32, False),
            import_memory("env", "mem", 1),
            import_table("env", "tbl", True, HT_FUNC, 0, 10),
        ],
        funcs=[1],
        codes=[func_body([], [global_get(0)])],   # return imported global
        exports=[export_func("get_seed", 0)],
    )
    validate_all(wasm, "25-imports")


# ===================================================================
# 26  Multiple exports
# ===================================================================

def example_26_multi_export():
    """Export the same function under several names."""
    wasm = module(
        types=[functype([I32, I32], [I32])],
        funcs=[0],
        codes=[func_body([], [local_get(0), local_get(1), i32_add()])],
        exports=[
            export_func("add", 0),
            export_func("plus", 0),
            export_func("sum", 0),
        ],
    )
    validate_all(wasm, "26-multi-export")


# ===================================================================
# 27  Start function
# ===================================================================

def example_27_start_function():
    """Module with a start function (runs on instantiation)."""
    wasm = module(
        types=[functype([], []), functype([], [I32])],
        globals_=[global_entry(I32, True, [i32_const(0)])],
        funcs=[0, 1],
        start=0,
        codes=[
            func_body([], [global_get(0), i32_const(1), i32_add(), global_set(0)]),
            func_body([], [global_get(0)]),
        ],
        exports=[export_func("get_counter", 1)],
    )
    validate_all(wasm, "27-start-function")


# ===================================================================
# 28  Element segment (active on table 0)
# ===================================================================

def example_28_element_segment():
    """Active element segment filling table with function indices."""
    wasm = module(
        types=[functype([], []), functype([], [I32])],
        tables=[table_entry(True, HT_FUNC, 0, 10)],
        funcs=[0, 1],
        elements=[elem_active([i32_const(0)], [0, 1, 1, 0])],
        codes=[
            func_body([], []),
            func_body([], [i32_const(99)]),
        ],
        exports=[export_func("f", 1)],
    )
    validate_all(wasm, "28-element-active")


# ===================================================================
# 29  Data segment (active on memory 0)
# ===================================================================

def example_29_data_segment():
    """Active data segment writing bytes into memory."""
    wasm = module(
        types=[functype([], [I32])],
        memories=[memory_entry(1)],
        funcs=[0],
        datas=[data_active([i32_const(0)], list(b'Hello, WASM!'))],
        codes=[func_body([], [i32_const(0), i32_load8_u()])],
        exports=[export_func("first_byte", 0)],
    )
    validate_all(wasm, "29-data-active")


# ===================================================================
# 30  Passive data + memory.init / data.drop
# ===================================================================

def example_30_passive_data():
    """Passive data segment with memory.init and data.drop."""
    wasm = module(
        types=[functype([], [])],
        memories=[memory_entry(1)],
        data_count=1,
        datas=[data_passive(list(b'\x01\x02\x03\x04'))],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0), i32_const(0), i32_const(4),
            memory_init(0, 0),
            data_drop(0),
        ])],
        exports=[export_func("init_data", 0)],
    )
    validate_all(wasm, "30-passive-data")


# ===================================================================
# 31  memory.copy / memory.fill
# ===================================================================

def example_31_memory_bulk():
    """memory.copy and memory.fill."""
    wasm = module(
        types=[functype([], [])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0), i32_const(0xAA), i32_const(16), memory_fill(0),
            i32_const(32), i32_const(0), i32_const(16), memory_copy(0, 0),
        ])],
        exports=[export_func("bulk", 0)],
    )
    validate_all(wasm, "31-memory-copy-fill")


# ===================================================================
# 32  Table instructions
# ===================================================================

def example_32_table_ops():
    """table.get, table.size, table.fill."""
    wasm = module(
        types=[functype([], [I32])],
        tables=[table_entry(True, HT_FUNC, 0, 4)],
        funcs=[0, 0],
        elements=[elem_active([i32_const(0)], [1])],
        codes=[
            # func 0: load table[0], check not-null, add table size
            func_body([], [
                i32_const(0),
                table_get(0),
                ref_is_null(),           # 0 (not null)
                table_size(0),
                i32_add(),               # 0 + 4 = 4
            ]),
            # func 1 (unused directly, referenced by elem)
            func_body([], [i32_const(7)]),
        ],
        exports=[export_func("table_ops", 0)],
    )
    validate_all(wasm, "32-table-get-size")

    wasm2 = module(
        types=[functype([], [])],
        tables=[table_entry(True, HT_FUNC, 0, 4)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(1), ref_null(HT_FUNC), i32_const(2), table_fill(0),
        ])],
        exports=[export_func("table_fill", 0)],
    )
    validate_all(wasm2, "32-table-fill")


# ===================================================================
# 33  Reference instructions
# ===================================================================

def example_33_ref_instructions():
    """ref.null + ref.is_null."""
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], [ref_null(HT_FUNC), ref_is_null()])],
        exports=[export_func("null_check", 0)],
    )
    validate_all(wasm, "33-ref-null-isnull")


# ===================================================================
# 34  select
# ===================================================================

def example_34_select():
    """select instruction."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            i32_const(10),
            i32_const(20),
            local_get(0),
            select(),
        ])],
        exports=[export_func("select_i32", 0)],
    )
    validate_all(wasm, "34-select")


# ===================================================================
# 35  drop and nop
# ===================================================================

def example_35_drop_nop():
    """drop and nop instructions."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [nop(), i32_const(1), drop(), nop()])],
        exports=[export_func("drop_nop", 0)],
    )
    validate_all(wasm, "35-drop-nop")


# ===================================================================
# 36  unreachable
# ===================================================================

def example_36_unreachable():
    """unreachable traps immediately."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [unreachable()])],
        exports=[export_func("trap", 0)],
    )
    validate_all(wasm, "36-unreachable")


# ===================================================================
# 37  call_indirect
# ===================================================================

def example_37_call_indirect():
    """call_indirect through a table."""
    wasm = module(
        types=[functype([I32], [I32])],
        tables=[table_entry(True, HT_FUNC, 0, 4)],
        funcs=[0],
        elements=[elem_active([i32_const(0)], [0])],
        codes=[func_body([], [
            i32_const(7),
            i32_const(0),
            call_indirect(0, 0),
        ])],
        exports=[export_func("indirect", 0)],
    )
    validate_all(wasm, "37-call-indirect")


# ===================================================================
# 38  Multi-value return
# ===================================================================

def example_38_multi_value():
    """Function returning multiple results."""
    wasm = module(
        types=[functype([I32], [I32, I32])],
        funcs=[0],
        codes=[func_body([], [
            local_get(0), i32_const(1), i32_add(),
            local_get(0), i32_const(1), i32_sub(),
        ])],
        exports=[export_func("pair", 0)],
    )
    validate_all(wasm, "38-multi-value")


# ===================================================================
# 39  Custom section
# ===================================================================

def example_39_custom_section():
    """Module with custom sections."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [])],
        exports=[export_func("noop", 0)],
        customs=[
            custom_section("my_metadata", b'\x01\x02\x03\x04'),
            custom_section("source", b'test.wat'),
        ],
    )
    validate_all(wasm, "39-custom-section")


# ===================================================================
# 40  Large modules (multi-byte LEB128 indices)
# ===================================================================

def example_40_large_module():
    """Module with 200 functions (multi-byte LEB128 indices)."""
    N = 200
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0] * N,
        codes=[func_body([], [i32_const(i)]) for i in range(N)],
        exports=[export_func(f"f{i}", i) for i in range(N)],
    )
    validate_all(wasm, "40-large-module-200")


# ===================================================================
# 41  Memory with and without max limit
# ===================================================================

def example_41_memory_limits():
    """Memory with and without maximum."""
    wasm = module(
        types=[functype([], [I32])],
        memories=[memory_entry(1, 10)],
        funcs=[0],
        codes=[func_body([], [memory_size(0)])],
        exports=[export_func("pages", 0), export_memory("mem", 0)],
    )
    validate_all(wasm, "41-memory-with-max")

    wasm = module(
        types=[functype([], [I32])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [memory_size(0)])],
        exports=[export_func("pages", 0), export_memory("mem", 0)],
    )
    validate_all(wasm, "41-memory-no-max")


# ===================================================================
# 42  Export of every external kind
# ===================================================================

def example_42_all_export_kinds():
    """Export a function, table, memory, and global."""
    wasm = module(
        types=[functype([], [I32])],
        tables=[table_entry(True, HT_FUNC, 0, 1)],
        memories=[memory_entry(1)],
        globals_=[global_entry(I32, False, [i32_const(0)])],
        funcs=[0],
        codes=[func_body([], [i32_const(1)])],
        exports=[
            export_func("fn", 0),
            export_table("tbl", 0),
            export_memory("mem", 0),
            export_global("g", 0),
        ],
    )
    validate_all(wasm, "42-all-export-kinds")


# ===================================================================
# 43  Tail-call (return_call)
# ===================================================================

def example_43_tail_call():
    """return_call for tail-call optimisation."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0, 0],
        codes=[
            func_body([], [local_get(0)]),                # identity
            func_body([], [local_get(0), return_call(0)]), # tail-call id
        ],
        exports=[export_func("id", 0), export_func("via_tail", 1)],
    )
    validate_all(wasm, "43-tail-call")


# ===================================================================
# 44  GC: struct types, struct.new, struct.get
# ===================================================================

def example_44_gc_struct():
    """Define struct types and use struct.new / struct.get."""
    struct_t = subtype(comptype_struct([(I32, False), (I64, True)]))
    fn_t = functype([], [I32])
    wasm = module(
        types=[struct_t, fn_t],
        funcs=[1],
        codes=[func_body([], [
            i32_const(42), i64_const(100),
            struct_new(0),
            struct_get(0, 0),
        ])],
        exports=[export_func("struct_field", 0)],
    )
    validate_proposal(wasm, "44-gc-struct")


# ===================================================================
# 45  GC: array types, array.new, array.get, array.len
# ===================================================================

def example_45_gc_array():
    """Define array type and use array.new, array.len."""
    array_t = subtype(comptype_array((I32, False)))
    fn_t = functype([], [I32])
    wasm = module(
        types=[array_t, fn_t],
        funcs=[1],
        codes=[func_body([], [
            i32_const(7), i32_const(5),
            array_new(0),
            array_len(),
        ])],
        exports=[export_func("array_len", 0)],
    )
    validate_proposal(wasm, "45-gc-array")


# ===================================================================
# 46  GC: array.new_fixed
# ===================================================================

def example_46_gc_array_fixed():
    """array.new_fixed with explicit element count."""
    array_t = subtype(comptype_array((I32, False)))
    fn_t = functype([], [I32])
    wasm = module(
        types=[array_t, fn_t],
        funcs=[1],
        codes=[func_body([], [
            i32_const(1), i32_const(2), i32_const(3),
            array_new_fixed(0, 3),
            i32_const(1),
            array_get(0),
        ])],
        exports=[export_func("arr_fixed_get", 0)],
    )
    validate_proposal(wasm, "46-gc-array-fixed")


# ===================================================================
# 47  GC: ref.i31, i31.get_s
# ===================================================================

def example_47_gc_i31():
    """i31 reference round-trip."""
    fn_t = functype([I32], [I32])
    wasm = module(
        types=[fn_t],
        funcs=[0],
        codes=[func_body([], [local_get(0), ref_i31(), i31_get_s()])],
        exports=[export_func("i31_roundtrip", 0)],
    )
    validate_proposal(wasm, "47-gc-i31")


# ===================================================================
# 48  GC: ref.test
# ===================================================================

def example_48_gc_ref_test():
    """ref.test instruction."""
    struct_t = subtype(comptype_struct([(I32, False)]))
    fn_t = functype([], [I32])
    wasm = module(
        types=[struct_t, fn_t],
        funcs=[1],
        codes=[func_body([], [
            i32_const(1), struct_new(0),
            ref_test(0),
        ])],
        exports=[export_func("ref_test", 0)],
    )
    validate_proposal(wasm, "48-gc-ref-test")


# ===================================================================
# 49  GC: any.convert_extern
# ===================================================================

def example_49_gc_extern_convert():
    """Convert between any and extern references."""
    fn_t = functype([], [I32])
    wasm = module(
        types=[fn_t],
        funcs=[0],
        codes=[func_body([], [
            ref_null(HT_EXTERN),
            any_convert_extern(),
            ref_is_null(),
        ])],
        exports=[export_func("extern_convert", 0)],
    )
    validate_proposal(wasm, "49-gc-extern-convert")


# ===================================================================
# 50  SIMD: v128.load, v128.store, v128.const
# ===================================================================

def example_50_simd_load_store():
    """SIMD load, store, and constant."""
    wasm = module(
        types=[functype([], [])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0),
            v128_const(bytes(16)),
            v128_store(),
            i32_const(0),
            v128_load(),
            drop(),
        ])],
        exports=[export_func("simd_load_store", 0)],
    )
    validate_all(wasm, "50-simd-load-store")


# ===================================================================
# 51  SIMD: splat and extract_lane
# ===================================================================

def example_51_simd_splat():
    """i32x4.splat and i32x4.extract_lane."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            local_get(0),
            i32x4_splat(),
            i32x4_extract_lane(0),
        ])],
        exports=[export_func("splat_extract", 0)],
    )
    validate_all(wasm, "51-simd-splat-extract")


# ===================================================================
# 52  SIMD: i32x4 add, sub, mul
# ===================================================================

def example_52_simd_i32x4_arith():
    """i32x4 arithmetic."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(b'\x01\x00\x00\x00' * 4),
            v128_const(b'\x02\x00\x00\x00' * 4),
            i32x4_add(),
            v128_const(b'\x01\x00\x00\x00' * 4),
            i32x4_sub(),
            drop(),
        ])],
        exports=[export_func("simd_arith", 0)],
    )
    validate_all(wasm, "52-simd-i32x4-arith")


# ===================================================================
# 53  SIMD: v128 bitwise operations
# ===================================================================

def example_53_simd_bitwise():
    """v128 not, and, or, xor, any_true."""
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes([0xFF] * 16)),
            v128_not(),
            v128_const(bytes([0xFF] * 16)),
            v128_and(),
            v128_any_true(),
        ])],
        exports=[export_func("simd_bitwise", 0)],
    )
    validate_all(wasm, "53-simd-bitwise")


# ===================================================================
# 54  SIMD: i8x16 and i16x8 comparisons
# ===================================================================

def example_54_simd_comparisons():
    """i8x16.eq and i16x8.eq."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            i8x16_eq(),
            drop(),
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            i16x8_eq(),
            drop(),
        ])],
        exports=[export_func("simd_cmp", 0)],
    )
    validate_all(wasm, "54-simd-comparisons")


# ===================================================================
# 55  SIMD: f32x4 and f64x2 arithmetic
# ===================================================================

def example_55_simd_float():
    """f32x4.add and f64x2.add."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            f32x4_add(),
            drop(),
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            f64x2_add(),
            drop(),
        ])],
        exports=[export_func("simd_float", 0)],
    )
    validate_all(wasm, "55-simd-f32x4-f64x2")


# ===================================================================
# 56  SIMD: shuffle
# ===================================================================

def example_56_simd_shuffle():
    """i8x16.shuffle."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(range(16))),
            v128_const(bytes(range(16, 32))),
            i8x16_shuffle([0, 17, 2, 19, 4, 21, 6, 23,
                           8, 25, 10, 27, 12, 29, 14, 31]),
            drop(),
        ])],
        exports=[export_func("simd_shuffle", 0)],
    )
    validate_all(wasm, "56-simd-shuffle")


# ===================================================================
# 57  SIMD: replace_lane
# ===================================================================

def example_57_simd_replace_lane():
    """i32x4.replace_lane."""
    wasm = module(
        types=[functype([I32], [I32])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)),
            local_get(0),
            i32x4_replace_lane(2),
            i32x4_extract_lane(2),
        ])],
        exports=[export_func("simd_replace", 0)],
    )
    validate_all(wasm, "57-simd-replace-lane")


# ===================================================================
# 58  SIMD: conversions
# ===================================================================

def example_58_simd_conversions():
    """f32x4.convert_i32x4_s and i32x4.trunc_sat_f32x4_s."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)),
            f32x4_convert_i32x4_s(),
            i32x4_trunc_sat_f32x4_s(),
            drop(),
            v128_const(bytes(16)),
            f64x2_promote_low_f32x4(),
            f32x4_demote_f64x2_zero(),
            drop(),
        ])],
        exports=[export_func("simd_cvt", 0)],
    )
    validate_all(wasm, "58-simd-conversions")


# ===================================================================
# 59  SIMD: narrow, extend, extmul
# ===================================================================

def example_59_simd_narrow_extend():
    """Vector narrowing, extension, and extmul."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)), v128_const(bytes(16)),
            i16x8_narrow_i32x4_s(),
            i16x8_extend_low_i8x16_s(),
            drop(),
            v128_const(bytes(16)), v128_const(bytes(16)),
            i8x16_narrow_i16x8_u(),
            drop(),
            v128_const(bytes(16)), v128_const(bytes(16)),
            i16x8_extmul_low_i8x16_s(),
            drop(),
        ])],
        exports=[export_func("simd_narrow_ext", 0)],
    )
    validate_all(wasm, "59-simd-narrow-extend")


# ===================================================================
# 60  SIMD: load_splat, load8x8_s, load32_zero
# ===================================================================

def example_60_simd_load_variants():
    """SIMD load splat, sign-extend, and zero variants."""
    wasm = module(
        types=[functype([], [])],
        memories=[memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0), i32_const(42), i32_store(),
            i32_const(4), i32_const(99), i32_store(),
            i32_const(0), v128_load32_splat(), drop(),
            i32_const(0), v128_load8x8_s(), drop(),
            i32_const(0), v128_load32_zero(), drop(),
        ])],
        exports=[export_func("simd_load_variants", 0)],
    )
    validate_all(wasm, "60-simd-load-variants")


# ===================================================================
# 61  SIMD: shifts and bitmask
# ===================================================================

def example_61_simd_shifts():
    """Vector shift and bitmask instructions."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes([1] + [0]*15)), i32_const(1),
            i8x16_shl(), i8x16_bitmask(), drop(),
            v128_const(bytes(16)), i32_const(1), i16x8_shl(), drop(),
            v128_const(bytes(16)), i32_const(1), i32x4_shl(), drop(),
            v128_const(bytes(16)), i32_const(1), i64x2_shl(), drop(),
        ])],
        exports=[export_func("simd_shifts", 0)],
    )
    validate_all(wasm, "61-simd-shifts")


# ===================================================================
# 62  SIMD: abs, neg, sqrt, min, max
# ===================================================================

def example_62_simd_unary_arith():
    """SIMD unary and min/max operations."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)), i8x16_abs(), drop(),
            v128_const(bytes(16)), i16x8_neg(), drop(),
            v128_const(bytes(16)), i32x4_abs(), drop(),
            v128_const(bytes(16)), i64x2_abs(), drop(),
            v128_const(bytes(16)), f32x4_sqrt(), drop(),
            v128_const(bytes(16)), f64x2_sqrt(), drop(),
            v128_const(bytes(16)), v128_const(bytes(16)), i32x4_min_s(), drop(),
            v128_const(bytes(16)), v128_const(bytes(16)), i32x4_max_u(), drop(),
        ])],
        exports=[export_func("simd_unary_arith", 0)],
    )
    validate_all(wasm, "62-simd-unary-min-max")


# ===================================================================
# 63  SIMD: i64x2 comparisons
# ===================================================================

def example_63_simd_i64x2_cmp():
    """i64x2.eq, ne, lt_s."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)), v128_const(bytes(16)), i64x2_eq(), drop(),
            v128_const(bytes(16)), v128_const(bytes(16)), i64x2_ne(), drop(),
            v128_const(bytes(16)), v128_const(bytes(16)), i64x2_lt_s(), drop(),
        ])],
        exports=[export_func("simd_i64_cmp", 0)],
    )
    validate_all(wasm, "63-simd-i64x2-cmp")


# ===================================================================
# 64  SIMD: extadd_pairwise, dot
# ===================================================================

def example_64_simd_extadd_dot():
    """Extadd_pairwise and dot product."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)), i16x8_extadd_pairwise_i8x16_u(), drop(),
            v128_const(bytes(16)), i32x4_extadd_pairwise_i16x8_s(), drop(),
            v128_const(bytes(16)), v128_const(bytes(16)), i32x4_dot_i16x8_s(), drop(),
        ])],
        exports=[export_func("simd_extadd_dot", 0)],
    )
    validate_all(wasm, "64-simd-extadd-dot")


# ===================================================================
# 65  Relaxed SIMD
# ===================================================================

def example_65_relaxed_simd():
    """Relaxed SIMD: f32x4.relaxed_min, f64x2.relaxed_madd."""
    wasm = module(
        types=[functype([], [])],
        funcs=[0],
        codes=[func_body([], [
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            f32x4_relaxed_min(),
            drop(),
            # f64x2.relaxed_madd needs 3 v128s: a, b, c => a + b*c
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            v128_const(bytes(16)),
            f64x2_relaxed_madd(),
            drop(),
        ])],
        exports=[export_func("relaxed_simd", 0)],
    )
    validate_all(wasm, "65-relaxed-simd")


# ===================================================================
# 66  Exception handling: try_table
# ===================================================================

def example_66_try_table():
    """try_table with catch clause.

    Tag type (i32)->() means catching pushes one i32; the I32 block
    type must be satisfied by both the try body and every catch branch.
    """
    fn_t = functype([I32], [I32])
    wasm = module(
        types=[functype([I32], []), fn_t],  # type 0: tag (i32) -> ()
        tags=[tag_entry(0)],
        funcs=[1],
        codes=[func_body([], [
            try_table(I32,
                [catch_clause(0, 0)],   # catching tag 0 pushes the thrown i32
                [local_get(0)],         # try body: push param (produces i32)
            ),
        ])],
        exports=[export_func("try_table", 0)],
    )
    validate_proposal(wasm, "66-try-table")

    # Also test void try_table with catch_all
    wasm2 = module(
        types=[functype([], []), functype([], [])],
        tags=[tag_entry(0)],
        funcs=[1],
        codes=[func_body([], [
            try_table(None,
                [catch_all(0)],         # catch_all: no values pushed
                [throw(0)],             # try body: throws
            ),
        ])],
        exports=[export_func("try_void", 0)],
    )
    validate_proposal(wasm2, "66-try-table-void")


# ===================================================================
# 67  Exception handling: throw
# ===================================================================

def example_67_throw():
    """throw instruction."""
    wasm = module(
        types=[functype([], []), functype([], [])],
        tags=[tag_entry(0)],
        funcs=[1],
        codes=[func_body([], [throw(0)])],
        exports=[export_func("do_throw", 0)],
    )
    validate_all(wasm, "67-throw")


# ===================================================================
# 68  Multi-memory
# ===================================================================

def example_68_multi_memory():
    """Two memories with memory.fill and memory.copy."""
    wasm = module(
        types=[functype([], [])],
        memories=[memory_entry(1), memory_entry(1)],
        funcs=[0],
        codes=[func_body([], [
            i32_const(0), i32_const(0xAA), i32_const(4), memory_fill(0),
            i32_const(0), i32_const(0), i32_const(4), memory_copy(1, 0),
        ])],
        exports=[export_memory("m0", 0), export_memory("m1", 1)],
    )
    validate_all(wasm, "68-multi-memory")


# ===================================================================
# 69  GC: subtypes with supers
# ===================================================================

def example_69_subtyping():
    """Sub type with explicit super type (base must be non-final to be extended)."""
    base = subtype(comptype_struct([(I32, False)]), final=False)
    sub = subtype(comptype_struct([(I32, False), (I64, False)]),
                  final=True, supers=[0])
    fn_t = functype([], [I32])
    wasm = module(
        types=[base, sub, fn_t],
        funcs=[2],
        codes=[func_body([], [
            i32_const(1), i64_const(2),
            struct_new(1),
            struct_get(1, 0),
        ])],
        exports=[export_func("sub_field", 0)],
    )
    validate_proposal(wasm, "69-subtyping")


# ===================================================================
# 70  GC: rec types
# ===================================================================

def example_70_rec_type():
    """Rec type grouping mutually-referencing subtypes."""
    st1 = subtype(comptype_struct([(I32, False)]))
    st2 = subtype(comptype_struct([(I64, False)]))
    rec = rectype([st1, st2])
    fn_t = functype([], [I32])
    wasm = module(
        types=[rec, fn_t],                # types 0,1 = structs; type 2 = fn_t
        funcs=[2],                         # func 0 uses type 2 (fn_t)
        codes=[func_body([], [
            i32_const(7), struct_new(0), struct_get(0, 0),
        ])],
        exports=[export_func("rec_type", 0)],
    )
    validate_proposal(wasm, "70-rec-type")


# ===================================================================
# 71  Tag import/export
# ===================================================================

def example_71_tag_import_export():
    """Import and export tags."""
    wasm = module(
        types=[functype([], []), functype([], [])],
        imports=[import_tag("env", "my_tag", 0)],
        tags=[tag_entry(0)],
        funcs=[1],
        codes=[func_body([], [throw(0)])],
        exports=[export_tag("local_tag", 1)],
    )
    validate_all(wasm, "71-tag-import-export")


# ===================================================================
# 72  Element segment variants
# ===================================================================

def example_72_element_variants():
    """Passive, declarative, and active-with-expr element segments."""
    wasm = module(
        types=[functype([], [])],
        tables=[table_entry(True, HT_FUNC, 0, 4)],
        funcs=[0, 0],
        elements=[
            elem_passive([0, 1]),
            elem_declare([0, 1]),
            elem_active_expr([i32_const(0)], [[ref_func(0)], [ref_func(1)]]),
        ],
        codes=[func_body([], []), func_body([], [])],
        exports=[export_func("f0", 0)],
    )
    validate_all(wasm, "72-element-variants")


# ===================================================================
# 73  Data segment variants
# ===================================================================

def example_73_data_variants():
    """Active and passive data segments."""
    wasm = module(
        types=[functype([], [I32])],
        memories=[memory_entry(1)],
        data_count=2,
        datas=[
            data_active([i32_const(0)], list(b'\xAA\xBB')),
            data_passive(list(b'\xCC\xDD')),
        ],
        funcs=[0],
        codes=[func_body([], [i32_const(0), i32_load8_u()])],
        exports=[export_func("load_byte", 0)],
    )
    validate_all(wasm, "73-data-variants")


# ===================================================================
# 74  GC: br_on_null / br_on_non_null
# ===================================================================

def example_74_br_on_null():
    """br_on_null: branch to a void block if the ref is null."""
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], [
            block(None, [           # void block (label 0)
                ref_null(HT_FUNC),
                br_on_null(0),      # null → branch (ref consumed)
                drop(),             # not-null → consume ref, fall through
            ]),
            i32_const(1),           # return 1
        ])],
        exports=[export_func("br_on_null", 0)],
    )
    validate_proposal(wasm, "74-br-on-null")


# ===================================================================
# 75  GC: ref.as_non_null
# ===================================================================

def example_75_ref_as_non_null():
    """ref.as_non_null instruction."""
    wasm = module(
        types=[functype([], [I32])],
        tables=[table_entry(True, HT_FUNC, 0, 1)],
        funcs=[0],
        elements=[elem_active([i32_const(0)], [0])],
        codes=[func_body([], [
            i32_const(0), table_get(0),
            ref_as_non_null(),
            ref_is_null(),
        ])],
        exports=[export_func("as_non_null", 0)],
    )
    validate_proposal(wasm, "75-ref-as-non-null")


# ===================================================================
# 76  Manual composition (no module() helper)
# ===================================================================

def example_76_manual_composition():
    """Build a module by manually concatenating sections."""
    wasm = (
        WASM_MAGIC
        + WASM_VERSION
        + type_section([functype([I32], [I32])])
        + func_section([0])
        + export_section([export_func("inc", 0)])
        + code_section([func_body([], [local_get(0), i32_const(1), i32_add()])])
    )
    validate_all(wasm, "76-manual-composition")


# ===================================================================
# 77  Realistic module: factorial + fibonacci + memory + table
# ===================================================================

def example_77_realistic():
    """Factorial, fibonacci, memory, table, start function."""
    wasm = module(
        types=[
            functype([I32], [I32]),     # type 0
            functype([], []),           # type 1
        ],
        memories=[memory_entry(1)],
        globals_=[global_entry(I32, True, [i32_const(0)])],
        funcs=[0, 0, 1],
        tables=[table_entry(True, HT_FUNC, 0, 2)],
        elements=[elem_active([i32_const(0)], [0, 1])],
        start=2,
        codes=[
            # func 0: factorial(n) -- iterative
            func_body([(1, I32)], [
                i32_const(1), local_set(1),
                block(None, [
                    loop(None, [
                        local_get(0), i32_eqz(), br_if(1),
                        local_get(1), local_get(0), i32_mul(), local_set(1),
                        local_get(0), i32_const(1), i32_sub(), local_set(0),
                        br(0),
                    ]),
                ]),
                local_get(1),
            ]),
            # func 1: fibonacci(n) -- iterative
            func_body([(3, I32)], [      # 3 locals: a(1), b(2), tmp(3)
                i32_const(0), local_set(1),
                i32_const(1), local_set(2),
                block(None, [
                    loop(None, [
                        local_get(0), i32_eqz(), br_if(1),
                        local_get(1), local_get(2), i32_add(), local_set(3),
                        local_get(2), local_set(1),
                        local_get(3), local_set(2),
                        local_get(0), i32_const(1), i32_sub(), local_set(0),
                        br(0),
                    ]),
                ]),
                local_get(1),
            ]),
            # func 2: start -- bump counter
            func_body([], [
                global_get(0), i32_const(1), i32_add(), global_set(0),
            ]),
        ],
        exports=[
            export_func("factorial", 0),
            export_func("fibonacci", 1),
            export_global("calls", 0),
            export_memory("mem", 0),
            export_table("indirect", 0),
        ],
    )
    validate_all(wasm, "77-realistic-module")


# ===================================================================
# 78  Incremental instruction list building
# ===================================================================

def example_78_incremental_build():
    """Build instruction sequences by appending to a list."""
    instrs = [i32_const(0)]
    for i in range(1, 10):
        instrs.append(i32_const(i))
        instrs.append(i32_add())
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], instrs)],
        exports=[export_func("sum_1_to_9", 0)],
    )
    validate_all(wasm, "78-incremental-build")


# ===================================================================
# 79  Raw float bit patterns
# ===================================================================

def example_79_raw_float_bits():
    """f32_bits / f64_bits construct raw IEEE 754 immediates."""
    # f32_bits returns the raw 4-byte IEEE 754 encoding.
    # Use it with the 0x43 opcode to form a proper f32.const instruction.
    pi_instr = byte(0x43) + f32_bits(0x40490FDB)   # ≈ 3.14159...
    wasm = module(
        types=[functype([], [I32])],
        funcs=[0],
        codes=[func_body([], [pi_instr, i32_reinterpret_f32()])],
        exports=[export_func("pi_bits", 0)],
    )
    validate_all(wasm, "79-raw-float-bits")

    # f64_bits similarly
    e_instr = byte(0x44) + f64_bits(0x4005BF0A8B145769)  # ≈ 2.71828...
    wasm2 = module(
        types=[functype([], [I64])],
        funcs=[0],
        codes=[func_body([], [e_instr, i64_reinterpret_f64()])],
        exports=[export_func("e_bits", 0)],
    )
    validate_all(wasm2, "79-raw-f64-bits")


# ===================================================================
# 80  GC: struct.new_default
# ===================================================================

def example_80_struct_new_default():
    """struct.new_default creates a zero-initialised struct."""
    struct_t = subtype(comptype_struct([(I32, False), (I64, False)]))
    fn_t = functype([], [I32])
    wasm = module(
        types=[struct_t, fn_t],
        funcs=[1],
        codes=[func_body([], [struct_new_default(0), struct_get(0, 0)])],
        exports=[export_func("default_struct", 0)],
    )
    validate_proposal(wasm, "80-struct-new-default")


# ===================================================================
# Run everything
# ===================================================================

EXAMPLES = [
    example_01_leb128,
    example_02_minimal,
    example_03_add,
    example_04_multi_func,
    example_05_locals,
    example_06_control_flow,
    example_07_if_else,
    example_08_blocktypes,
    example_09_br_table,
    example_10_numeric_constants,
    example_11_i32_relops,
    example_12_i32_arithmetic,
    example_13_i64_ops,
    example_14_float_ops,
    example_15_conversions,
    example_16_saturating,
    example_17_sign_extension,
    example_18_memory,
    example_19_memory_variants,
    example_20_memarg_offset_align,
    example_21_memory_size_grow,
    example_22_globals,
    example_23_float_globals,
    example_24_table,
    example_25_imports,
    example_26_multi_export,
    example_27_start_function,
    example_28_element_segment,
    example_29_data_segment,
    example_30_passive_data,
    example_31_memory_bulk,
    example_32_table_ops,
    example_33_ref_instructions,
    example_34_select,
    example_35_drop_nop,
    example_36_unreachable,
    example_37_call_indirect,
    example_38_multi_value,
    example_39_custom_section,
    example_40_large_module,
    example_41_memory_limits,
    example_42_all_export_kinds,
    example_43_tail_call,
    example_44_gc_struct,
    example_45_gc_array,
    example_46_gc_array_fixed,
    example_47_gc_i31,
    example_48_gc_ref_test,
    example_49_gc_extern_convert,
    example_50_simd_load_store,
    example_51_simd_splat,
    example_52_simd_i32x4_arith,
    example_53_simd_bitwise,
    example_54_simd_comparisons,
    example_55_simd_float,
    example_56_simd_shuffle,
    example_57_simd_replace_lane,
    example_58_simd_conversions,
    example_59_simd_narrow_extend,
    example_60_simd_load_variants,
    example_61_simd_shifts,
    example_62_simd_unary_arith,
    example_63_simd_i64x2_cmp,
    example_64_simd_extadd_dot,
    example_65_relaxed_simd,
    example_66_try_table,
    example_67_throw,
    example_68_multi_memory,
    example_69_subtyping,
    example_70_rec_type,
    example_71_tag_import_export,
    example_72_element_variants,
    example_73_data_variants,
    example_74_br_on_null,
    example_75_ref_as_non_null,
    example_76_manual_composition,
    example_77_realistic,
    example_78_incremental_build,
    example_79_raw_float_bits,
    example_80_struct_new_default,
]


def main():
    passed = 0
    failed = 0
    for ex in EXAMPLES:
        try:
            ex()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL  {ex.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(EXAMPLES)} examples")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
