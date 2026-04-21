// WAM Runtime in LP Form
// Translated from wam_wasm.py
//
// Tags: REF=0, CON=1, STR=2, LIS=3, FUN=4
// Heap cells are 2 i32 slots: [tag, value]

// --- Globals ---
global H = 0.
global E = 0.
global B = -1.
global HB = 0.
global TR = 0.
global S = 0.
global MODE = 0.
global FAIL = 0.
global HALTED = 0.
global ESTACK = 0.
global PDL_TOP = 0.
global CONT_TOP = 0.
global BP_TOP = 0.

// --- Arrays ---
array HEAP.
array TRAIL.
array STACK.
array PDL.
array XREG.
array CONT: ref.
array BP_STACK: ref.

// ===================================================================
// init — create arrays and reset registers
// ===================================================================
init(;):
    anew(HEAP, 65536;),
    anew(TRAIL, 16384;),
    anew(STACK, 16384;),
    anew(PDL, 4096;),
    anew(XREG, 256;),
    rnew(CONT, 16384;),
    rnew(BP_STACK, 4096;),
    gset(B, -1;),
    gset(BP_TOP, 0;),
    gset(H, 0;),
    gset(E, 0;),
    gset(ESTACK, 0;),
    gset(TR, 0;),
    gset(FAIL, 0;).

// ===================================================================
// heap_push(tag, val; addr) — push a cell, return old H
// ===================================================================
heap_push(tag, val; old_h):
    gget(H; old_h),
    mul(old_h, 2; idx),
    aset(HEAP, idx, tag;),
    add(idx, 1; idx1),
    aset(HEAP, idx1, val;),
    add(old_h, 1; new_h),
    gset(H, new_h;).

// ===================================================================
// heap_get_tag(addr; tag)
// ===================================================================
heap_get_tag(addr; tag):
    mul(addr, 2; idx),
    aget(HEAP, idx; tag).

// ===================================================================
// heap_get_val(addr; val)
// ===================================================================
heap_get_val(addr; val):
    mul(addr, 2; idx),
    add(idx, 1; idx1),
    aget(HEAP, idx1; val).

// ===================================================================
// heap_set_tag(addr, tag;)
// ===================================================================
heap_set_tag(addr, tag;):
    mul(addr, 2; idx),
    aset(HEAP, idx, tag;).

// ===================================================================
// heap_set_val(addr, val;)
// ===================================================================
heap_set_val(addr, val;):
    mul(addr, 2; idx),
    add(idx, 1; idx1),
    aset(HEAP, idx1, val;).

// ===================================================================
// deref(addr; ret) — chase REF chain
// ===================================================================
// Non-REF: return immediately
deref(addr; ret):
    heap_get_tag(addr; tag), tag != 0,
    copy(addr; ret).
// REF self-ref (unbound): return
deref(addr; ret):
    heap_get_tag(addr; tag), tag == 0,
    heap_get_val(addr; val), val == addr,
    copy(addr; ret).
// REF chain: follow
deref(addr; ret):
    heap_get_tag(addr; tag), tag == 0,
    heap_get_val(addr; val), val != addr,
    deref(val; ret).

// ===================================================================
// trail_check(addr;) — conditionally trail an address
// ===================================================================
trail_check(addr;):
    gget(HB; hb), addr < hb,
    gget(TR; tr),
    aset(TRAIL, tr, addr;),
    add(tr, 1; new_tr),
    gset(TR, new_tr;).
trail_check(addr;).

// ===================================================================
// bind(a1, a2;) — bind two heap cells with trailing
// ===================================================================
// a1 is REF, a2 is not REF: bind a1 toward a2
bind(a1, a2;):
    heap_get_tag(a1; tag1), tag1 == 0,
    heap_get_tag(a2; tag2), tag2 != 0,
    heap_get_val(a2; v2),
    heap_set_tag(a1, tag2;),
    heap_set_val(a1, v2;),
    trail_check(a1;).
// Both REF, a2 < a1: bind a1 toward a2
bind(a1, a2;):
    heap_get_tag(a1; tag1), tag1 == 0,
    heap_get_tag(a2; tag2), tag2 == 0,
    a2 < a1,
    heap_get_val(a2; v2),
    heap_set_tag(a1, tag2;),
    heap_set_val(a1, v2;),
    trail_check(a1;).
// Fallback: bind a2 toward a1
bind(a1, a2;):
    heap_get_tag(a1; t1),
    heap_get_val(a1; v1),
    heap_set_tag(a2, t1;),
    heap_set_val(a2, v1;),
    trail_check(a2;).

// ===================================================================
// pdl_push(val;) / pdl_pop(; val)
// ===================================================================
pdl_push(val;):
    gget(PDL_TOP; pt),
    aset(PDL, pt, val;),
    add(pt, 1; new_pt),
    gset(PDL_TOP, new_pt;).

pdl_pop(; val):
    gget(PDL_TOP; pt),
    sub(pt, 1; new_pt),
    gset(PDL_TOP, new_pt;),
    aget(PDL, new_pt; val).

// ===================================================================
// unify(a1, a2;) — full PDL-based unification
// ===================================================================
unify(a1, a2;):
    gset(PDL_TOP, 0;),
    gset(FAIL, 0;),
    pdl_push(a1;),
    pdl_push(a2;),
    unify_loop(;).

// --- Main loop ---
unify_loop(;):
    gget(PDL_TOP; pt), pt == 0.
unify_loop(;):
    gget(FAIL; f), f != 0.
unify_loop(;):
    gget(PDL_TOP; pt), pt != 0,
    gget(FAIL; f), f == 0,
    pdl_pop(; raw1), deref(raw1; d1),
    pdl_pop(; raw2), deref(raw2; d2),
    unify_pair(d1, d2;),
    unify_loop(;).

// --- Pair dispatch ---
// Same address: skip
unify_pair(d1, d2;): d1 == d2.
// tag1 == REF: bind
unify_pair(d1, d2;):
    d1 != d2,
    heap_get_tag(d1; tag1), tag1 == 0,
    bind(d1, d2;).
// tag2 == REF: bind reversed
unify_pair(d1, d2;):
    d1 != d2,
    heap_get_tag(d1; tag1), tag1 != 0,
    heap_get_tag(d2; tag2), tag2 == 0,
    bind(d2, d1;).
// CON + CON: compare values
unify_pair(d1, d2;):
    d1 != d2,
    heap_get_tag(d1; tag1), tag1 == 1,
    heap_get_tag(d2; tag2), tag2 == 1,
    heap_get_val(d1; val1),
    heap_get_val(d2; val2),
    unify_con(val1, val2;).
// LIS + LIS: push car/cdr pairs
unify_pair(d1, d2;):
    d1 != d2,
    heap_get_tag(d1; tag1), tag1 == 3,
    heap_get_tag(d2; tag2), tag2 == 3,
    heap_get_val(d1; v1),
    heap_get_val(d2; v2),
    pdl_push(v1;), pdl_push(v2;),
    add(v1, 1; v1b), add(v2, 1; v2b),
    pdl_push(v1b;), pdl_push(v2b;).
// STR + STR: compare functors, push arg pairs
unify_pair(d1, d2;):
    d1 != d2,
    heap_get_tag(d1; tag1), tag1 == 2,
    heap_get_tag(d2; tag2), tag2 == 2,
    heap_get_val(d1; v1),
    heap_get_val(d2; v2),
    unify_str(v1, v2;).
// Default: type mismatch → fail
unify_pair(d1, d2;):
    gset(FAIL, 1;).

// --- CON comparison ---
unify_con(v1, v2;): v1 == v2.
unify_con(v1, v2;): v1 != v2, gset(FAIL, 1;).

// --- STR comparison ---
unify_str(v1, v2;):
    heap_get_val(v1; packed1),
    heap_get_val(v2; packed2),
    unify_str_check(v1, v2, packed1, packed2;).

unify_str_check(v1, v2, p1, p2;):
    p1 != p2,
    gset(FAIL, 1;).
unify_str_check(v1, v2, p1, p2;):
    p1 == p2,
    and(p1, 255; arity),
    unify_str_args(v1, v2, 1, arity;).

unify_str_args(v1, v2, i, arity;): i > arity.
unify_str_args(v1, v2, i, arity;):
    i <= arity,
    add(v1, i; a1), add(v2, i; a2),
    pdl_push(a1;), pdl_push(a2;),
    add(i, 1; i'),
    unify_str_args(v1, v2, i', arity;).

// ===================================================================
// backtrack_restore(;) — restore state from current choice point
// ===================================================================
backtrack_restore(;):
    gget(B; b),
    aget(STACK, b; saved_e),
    gset(E, saved_e;),
    add(b, 2; b2),
    aget(STACK, b2; saved_tr),
    add(b, 3; b3),
    aget(STACK, b3; saved_h),
    unwind_trail(saved_tr;),
    gset(H, saved_h;),
    gset(HB, saved_h;),
    gset(TR, saved_tr;),
    add(b, 4; b4),
    aget(STACK, b4; n),
    restore_xregs(b, 1, n;),
    gset(FAIL, 0;).

// --- Trail unwinding ---
unwind_trail(saved_tr;):
    gget(TR; tr), tr <= saved_tr.
unwind_trail(saved_tr;):
    gget(TR; tr), tr > saved_tr,
    sub(tr, 1; new_tr),
    gset(TR, new_tr;),
    aget(TRAIL, new_tr; addr),
    mul(addr, 2; idx),
    aset(HEAP, idx, 0;),
    add(idx, 1; idx1),
    aset(HEAP, idx1, addr;),
    unwind_trail(saved_tr;).

// --- X register restoration ---
restore_xregs(b, i, n;): i > n.
restore_xregs(b, i, n;):
    i <= n,
    add(b, 4; b4),
    add(b4, i; bi),
    aget(STACK, bi; val),
    aset(XREG, i, val;),
    add(i, 1; i'),
    restore_xregs(b, i', n;).

// ===================================================================
// Prolog front-end helpers
// ===================================================================
// These are the primitive building blocks used by prolog_to_lp.py
// when lowering a Prolog program. Each Prolog clause becomes an LP
// procedure that takes one heap address per argument and composes
// the primitives below to match head and execute body.

// put_variable(; addr) — allocate a fresh unbound variable cell
// (REF pointing to itself), return its heap address.
put_variable(; addr):
    gget(H; addr),
    heap_push(0, addr; _).

// put_constant(val; addr) — allocate a constant cell (CON/val),
// return its heap address.
put_constant(val; addr):
    heap_push(1, val; addr).

// match_constant(arg, val;) — unify heap cell `arg` with CON(val).
// Sets FAIL=1 if unification fails. Skipped if FAIL is already set.
match_constant(arg, val;):
    gget(FAIL; f), f != 0.
match_constant(arg, val;):
    gget(FAIL; f), f == 0,
    heap_push(1, val; c),
    unify(arg, c;).

// ===================================================================
// test(; result) — unify f(X) with f(42), return X = 42
// ===================================================================
test(; result):
    init(;),
    // Build f(X): FUN(f/1) at 0, REF(1) at 1, STR(0) at 2
    heap_push(4, 257; _a0),
    heap_push(0, 1; _a1),
    heap_push(2, 0; _a2),
    // Build f(42): FUN(f/1) at 3, CON(42) at 4, STR(3) at 5
    heap_push(4, 257; _a3),
    heap_push(1, 42; _a4),
    heap_push(2, 3; _a5),
    // unify(2, 5)
    unify(2, 5;),
    // result = heap_get_val(deref(1))
    deref(1; d),
    heap_get_val(d; result).
