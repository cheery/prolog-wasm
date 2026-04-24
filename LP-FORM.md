# LP Form Language Specification

LP Form is a restricted, deterministic logic-programming language based on
Gange et al. (TPLP 2015). A program is a set of Horn clauses where each
procedure has complementary guards ensuring exactly one clause fires per call.
The same source compiles to WebAssembly and extracts to Constrained Horn
Clauses for formal verification.

## Lexical Structure

**Identifiers** match `[a-zA-Z_][a-zA-Z0-9_]*'*` — letters, digits,
underscores, and optionally trailing primes (e.g. `b'`, `acc'`,
`n_prime'`).

**Integer literals** match `-?[0-9]+`.

**Comparisons**: `==`, `!=`, `<`, `<=`, `>`, `>=`.

**Punctuation**: `(`, `)`, `{`, `}`, `;`, `,`, `.`, `:`, `=`, `|`.

**Comments**:
- Line comments: `//` to end of line.
- Block comments: `/* ... */` (non-nesting).

Whitespace between tokens is ignored.

## Program Structure

A program is a sequence of declarations followed by clauses:

```
program ::= decl* clause+
decl    ::= global_decl | array_decl | struct_decl | type_decl | adt_decl
```

Declarations must precede all clauses. Clauses with the same head name are
grouped into a single procedure.

## Declarations

### Global Variables

```
global_decl ::= "global" NAME "=" SIGNED_NUMBER "."
```

Declares a mutable i32 global with an initial value. Example:

```
global H = 0.
global B = -1.
```

### Arrays

```
array_decl ::= "array" NAME "."
             | "array" NAME ":" "ref" "."
```

Declares a named mutable array. Default kind is `i32`. With `: ref` the
array holds function references. Example:

```
array HEAP.
array CONT: ref.
```

### Struct Types

```
struct_decl ::= "struct" NAME "{" field_list "}" "."
field_list  ::= field ("," field)*
field       ::= NAME ":" NAME
```

Declares a struct type with named fields. All fields are `i32`. Example:

```
struct Cell { tag: i32, val: i32 }.
```

Structs are allocated with `struct_new` and fields accessed with dot
notation or `struct_get`.

### Sum Types

```
type_decl  ::= "type" NAME "=" ctor_list "."
ctor_list  ::= ctor ("|" ctor)*
ctor       ::= NAME "(" type_list ")" | NAME
type_list  ::= NAME ("," NAME)*
```

Declares a sum type (tagged union) with named constructors. Each
constructor carries a positional payload. All payload types are `i32`.
Example:

```
type Cell = ref(i32) | con(i32) | fun(i32, i32).
```

Sum types are represented as GC structs with a `__tag` field (i32) followed
by payload fields `_f0`, `_f1`, etc., sized for the largest constructor.
Constructors are used as patterns in call output positions.

### Abstract Data Types

```
adt_decl   ::= "adt" NAME "{" sig_list "}" "."
sig_list   ::= sig*
sig        ::= NAME "(" in_params ";" sig_out_params ")"
             | NAME "(" ";" sig_out_params ")"
             | NAME "(" in_params ";" ")"
             | NAME "(" ";" ")"
```

Declares an ADT — a named interface whose signatures must be implemented by
procedures of the same name and arity. ADT blocks are descriptive; they do
not affect code generation. Example:

```
adt Counter {
    init(n; c)
    inc(c; c_new)
    get(c; v)
}.
```

## Procedures and Clauses

### Clause Syntax

```
clause ::= head ":" body "."
         | head "."
head   ::= NAME "(" in_params ";" out_vals ")"
         | NAME "(" ";" out_vals ")"
         | NAME "(" in_params ";" ")"
         | NAME "(" ";" ")"
body   ::= goal ("," goal)*
```

A clause has a head and an optional body. The head names a procedure,
declares input parameters (before `;`) and output parameters (after `;`).
The body is a comma-separated sequence of goals.

All clauses with the same head name belong to the same procedure and must
share the same input/output arity. Within a procedure, clauses are tried in
order; the first clause whose guards hold fires.

### Input Parameters

```
in_params ::= NAME ("," NAME)*
```

Input parameters are variable names that are bound on entry to the clause.

### Output Values

```
out_vals  ::= val ("," val)*
val       ::= NAME
            | SIGNED_NUMBER
            | NAME "." NAME
```

Output values in the head can be variables, constants, or field accesses.
If a head output is an input variable or a constant, the parser inserts an
implicit `copy` goal.

Example — the second clause uses `a` (an input) as output, which the parser
expands to `copy(a; _ret0)`:

```
gcd(a, b; ret): b != 0, mod(a, b; b'), gcd(b, b'; ret).
gcd(a, b; a):   b == 0.
```

## Values

| Form | Meaning | Example |
|---|---|---|
| `NAME` | Variable reference | `x`, `acc'`, `b_prime` |
| `SIGNED_NUMBER` | Integer constant | `0`, `42`, `-1` |
| `NAME.NAME` | Field access on a struct/sum value | `c.tag`, `p.x` |

Field access is resolved during elaboration into a `struct_get` primop with
a fresh temporary variable.

## Goals

A goal is one of: a guard, a primop, or a procedure call.

### Guards

```
guard ::= val CMP val
```

A guard is an inline comparison that must hold for the clause to fire.
Guards produce no outputs. The comparison operators are:

| Syntax | Internal | Meaning |
|---|---|---|
| `==` | `eq` | Equal |
| `!=` | `ne` | Not equal |
| `<` | `lt` | Signed less than |
| `<=` | `le` | Signed less or equal |
| `>` | `gt` | Signed greater than |
| `>=` | `ge` | Signed greater or equal |

Example: `b != 0`, `n > 0`, `x == 0`.

### PrimOps

```
call_goal ::= NAME "(" args_io ")"
            | NAME "(" ")"
            | NAME "(" ";" ")"

args_io   ::= vals ";" out_names
            | vals ";"
            | ";" out_names
            | vals
```

Built-in primitive operations are syntactically identical to procedure calls
but are recognized by name. They are listed in the [Built-in Operations](#built-in-operations)
section below.

### Procedure Calls

```
call_goal ::= NAME "(" args_io ")"
```

A call to a user-defined procedure. Inputs are values (variables or
constants); outputs are variable names or constructor patterns. Example:

```
gcd(b, b'; ret)
fact_acc(n, 1; ret)
```

### Output Names and Patterns

```
out_names ::= out_name ("," out_name)*
out_name  ::= NAME "(" out_var_list ")"   -- constructor pattern
            | NAME                          -- plain variable

out_var_list ::= NAME ("," NAME)* | /* empty */
```

Outputs can be plain variable names or constructor patterns that
destructure a sum-type value. See [Pattern Matching](#pattern-matching).

## Built-in Operations

| Operation | Signature | Description |
|---|---|---|
| `add(a, b; r)` | `(i32, i32; i32)` | `r = a + b` |
| `sub(a, b; r)` | `(i32, i32; i32)` | `r = a - b` |
| `mul(a, b; r)` | `(i32, i32; i32)` | `r = a * b` |
| `div(a, b; r)` | `(i32, i32; i32)` | `r = a / b` (signed) |
| `rem(a, b; r)` | `(i32, i32; i32)` | `r = a % b` (signed). `mod` is an alias. |
| `copy(a; r)` | `(i32; i32)` | `r = a` |
| `neg(a; r)` | `(i32; i32)` | `r = -a` |
| `and(a, b; r)` | `(i32, i32; i32)` | Bitwise AND |
| `or(a, b; r)` | `(i32, i32; i32)` | Bitwise OR |
| `gget(NAME; r)` | `(name; i32)` | Read global `NAME` into `r` |
| `gset(NAME, v;)` | `(name, i32;)` | Set global `NAME` to `v` |
| `aget(ARR, i; r)` | `(name, i32; i32)` | Read element `i` from array `ARR` |
| `aset(ARR, i, v;)` | `(name, i32, i32;)` | Write `v` to index `i` of array `ARR` |
| `anew(ARR, n;)` | `(name, i32;)` | Allocate i32 array of size `n`, bind to `ARR` |
| `rnew(ARR, n;)` | `(name, i32;)` | Allocate funcref array of size `n`, bind to `ARR` |
| `struct_new(TYPE, v1, ...; r)` | `(name, i32, ...; ref)` | Allocate struct of type `TYPE` with field values |
| `struct_get(val; r)` | `(ref; i32)` | Read a struct field. Used after elaboration with metadata. |

For `gget`, `gset`, `aget`, `aset`, `anew`, `rnew`, and `struct_new`, the
first argument is a global or array name (not a variable).

## Pattern Matching

Constructor patterns appear in call output positions to destructure sum-type
values:

```
kind(a; k):
    source(a; ref(t)), copy(1; k).
kind(a; k):
    source(a; con(v)), copy(2; k).
```

Here `ref(t)` and `con(v)` are constructor patterns. The call `source(a;
ref(t))` means: call `source(a; tmp)`, then check that `tmp`'s tag matches
`ref` (tag 0) and bind the first payload field to `t`.

### Exhaustiveness

All clauses in a procedure that use pattern matching must:

1. Dispatch on the **same** call and **same** output position.
2. Cover **every** constructor of the sum type at least once.
3. Not mix pattern clauses with plain-variable clauses.
4. Not use wildcard patterns (`_`) in the dispatch position.

Multiple clauses per constructor are allowed (disambiguated by further
guards on the payload variables).

### Elaboration

Pattern matching is resolved by the elaboration pass. For each constructor
pattern `Ctor(v1, v2, ...)`, the elaborator generates:

1. A fresh temporary to receive the call output.
2. A `struct_get` to read the `__tag` field.
3. A guard `tag == N` where `N` is the constructor's index.
4. A `struct_get` for each payload field, bound to `v1`, `v2`, etc.

After elaboration, no `LPPattern` or `LPFieldAccess` nodes remain.

## Tail Calls

A call is a tail call when it is the **last goal** in a clause and its
outputs match the clause head's outputs exactly. Tail calls are detected
automatically and emitted as WASM `return_call`, giving constant stack usage
for tail-recursive loops.

Example — `gcd(b, b'; ret)` is a tail call in:

```
gcd(a, b; ret): b != 0, mod(a, b; b'), gcd(b, b'; ret).
```

## Validation Rules

The validator checks the following invariants:

1. **Matching signatures**: All clauses in a procedure share the same head
   name, input arity, and output arity.

2. **Single assignment**: Each variable is assigned at most once per clause.
   Assignment occurs via primop outputs, call outputs, or pattern variable
   bindings. Input parameters are considered assigned on entry.

3. **Output definition**: Every output variable named in the clause head must
   be assigned by some goal in the body.

4. **No duplicate procedures**: Each procedure name is unique.

5. **ADT signature matching**: Every signature in an ADT block has a
   corresponding procedure with matching input and output arities.

## Compilation Pipeline

The pipeline `lp_compile` in `lp_pipeline.py` runs these passes in order:

1. **Validate** — structural checks (signatures, single assignment, ADTs).
2. **Infer output types** — determine which procedure outputs hold struct/sum
   references (for WASM ref-typed returns).
3. **Elaborate** — resolve patterns and field accesses into core IR.
4. **Mark tail calls** — identify tail-position calls.
5. **Emit** — generate WASM module bytes.

## CHC Extraction

Because LP Form clauses are Horn clauses, extraction to SMT-LIB2 is
mechanical:

- Each procedure becomes an uninterpreted predicate.
- Each clause becomes a `(forall ... body => head)` assertion.
- Clause ordering (first-match-wins) is encoded by negating prior clauses'
  guard preambles.
- Globals and arrays are threaded through predicates via a state-passing
  transform (pre/post pairs).
- i32 globals map to `Int`; i32 arrays map to `(Array Int Int)`.
- Ref arrays use opaque uninterpreted sorts (sound but imprecise).
- Struct and sum types use uninterpreted `Int`-sorted constructors and
   field accessor axioms.

## Complete Example

```
// Euclid's GCD
gcd(a, b; ret): b != 0, rem(a, b; b'), gcd(b, b'; ret).
gcd(a, b; a):   b == 0.
```

With state:

```
global H = 0.
array HEAP.

heap_push(tag, val; old_h):
    gget(H; old_h),
    mul(old_h, 2; idx),
    aset(HEAP, idx, tag;),
    add(idx, 1; idx1),
    aset(HEAP, idx1, val;),
    add(old_h, 1; new_h),
    gset(H, new_h;).
```

With structs and sum types:

```
struct Cell { tag: i32, val: i32 }.

type Option = some(i32) | none.

unwrap_or(opt, default; out):
    get_opt(opt; some(v)), copy(v; out).
unwrap_or(opt, default; out):
    get_opt(opt; none()), copy(default; out).

get_opt(x; r): copy(x; r).
```
