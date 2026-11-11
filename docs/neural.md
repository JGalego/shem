# The neural vCPU

The neural vCPU ([`neural/`](../neural/)) is an RV32IM CPU compiled into the weights of a network. One forward pass executes one instruction, for every guest in a batch at once. It implements the same [`Vcpu` interface](../crates/shem/src/vcpu.rs) as the reference interpreter, and the host drives it over a Unix socket, so containers run on it unchanged: `shem run --vcpu neural`.

It follows the idea behind [Qwenputer](https://divagr.com/blog/qwenputer), compiled layers for the CPU and DeltaNet-style state for memory, but it is a standalone network rather than a CPU grafted into a pretrained language model.

## Compiled, not trained

[`isa.py`](../neural/src/shem_neural/isa.py) describes the instruction set as data: for each of the 48 RV32IM instructions, a bit pattern (mask and match) and what it does, in terms of which unit's result goes to `rd`, where its immediate comes from, how the next pc is chosen, and whether it loads, stores or exits. [`compile.py`](../neural/src/shem_neural/compile.py) turns that table into the network's tensors:

| Tensor | Shape | Role |
|---|---|---|
| `decode_w`, `decode_c` | 48 × 32, 48 | A layer that counts, for each instruction, how many masked bits of the word differ from its pattern. `relu(1 − count)` is a one-hot over instructions, and all zeros for an illegal one. |
| `index_w`, `index_c` | 32 × 5, 32 | The same trick for 5-bit register numbers and shift amounts |
| `imm_bits` | 5 × 32 × 32 | Selection matrices from instruction bits to immediate bits, one per format, sign extension included |
| `format_route`, `result_route`, `next_pc_route`, `branch_route`, `exit_route` | 48 × n | Routing tables: which format, result, next pc, comparison and exit each instruction uses |
| `uses_imm`, `load_size`, `store_size` | 48 | Per-instruction flags and sizes |
| `sll`, `srl`, `sra_fill` | 32 × 32 × 32 | One shift matrix per shift amount; the shifter is attention over them |
| `byte_bits`, `powers` | 256 × 8, 32 | Byte-to-bits lookup table, and the weights that turn bits into a value |

The weights can be saved as safetensors, the format a machine image carries them in (see [shemfile.md](shemfile.md)).

## State

| Part | Shape | Representation |
|---|---|---|
| Register file | B × 32 × 32 | A delta-rule state with one row of bits per register |
| Memory | B × mem_size | A delta-rule state with one row per byte |
| pc | B, plus its two low bits | The pc's low bits drive the alignment check |
| instret | B | Instructions retired |
| base | B | The guest address where memory starts: 0 for containers, `0x80000000` in VM mode |

Both the register file and memory are written with the delta rule, `S ← S + β (v − S k) kᵀ`, with a one-hot key `k`. With β = 1 that replaces the keyed row exactly; with β = 0 it changes nothing. A read is `S q` with a one-hot query `q`, which returns the row. This is how Gated DeltaNet layers hold memory in Qwenputer.

## One forward pass

[`cpu.py`](../neural/src/shem_neural/cpu.py), in order:

1. **Fetch.** Four one-hot reads of memory at the pc's offset from the base give the instruction's bytes, and a table lookup turns them into 32 bits. A clamp flags a pc that is misaligned or out of range. The offset is `pc − base + 2³² · clamp(base − pc)`: below the base it wraps past the end of memory, so it faults, as on the reference vCPU. Loads and stores use the same offset.
2. **Decode.** The mismatch-counting layer gives the instruction one-hot. Three more give one-hots for `rd`, `rs1` and `rs2`.
3. **Operands.** Attention reads of the register file with the `rs1` and `rs2` one-hots give the operands' bits. The selection matrices give the immediate's bits. The second operand is the immediate or `rs2`, chosen by `uses_imm`.
4. **Units.** Every unit computes its result for every guest: adder, logic (`a·b`, `a + b − a·b`, `a + b − 2a·b` on bits), shifter (attention over the shift matrices, with sign fill for `sra`), comparator, load extractor, multiplier (on 16-bit halves, so every partial value stays exact) and divider (restoring division, 32 unrolled compare-and-subtract layers). Values are integers, linear in the bits. A stack of threshold layers, `decompose`, turns a value back into bits.
5. **Exits.** ecall and ebreak come from the decode, illegal is "no instruction matched", and memory faults compare the address against the memory size. They are combined in the reference vCPU's order: fetch, then decode, then execute.
6. **Routing.** The instruction one-hot times the routing tables picks the result for `rd`, the comparison that decides a branch, and the next pc.
7. **Writeback.** Delta-rule writes to the register file and memory, and the pc update, all gated by `commit`, which is 0 when the instruction exits. An exiting instruction therefore changes nothing, as on the reference vCPU.

There is no data-dependent control flow in the arithmetic: the decoded instruction selects among results that were all computed. So guests running different instructions cost the same as guests running the same one, which is what lets one batch carry unrelated guests (design rule 2 in [architecture.md](architecture.md)). A test runs three different programs as one batch and checks that each ends exactly as it does alone.

## The operations it uses

Only operations a compiled transformer has: products with the compiled weights (linear layers and one-hot attention), element-wise sums and products (gating, as in gated MLPs), and clamps, which are piecewise-linear like ReLU. State changes only through gated delta-rule writes.

Values are integers held exactly in float64. Qwenputer runs in bf16, which forces arithmetic on small values; float64 lets a 32-bit word be one exact value, which keeps the network simple. A bf16 version would work on bytes instead of words.

## Two executors

The network as described, the **dense** executor, is the definition. The **fused** executor computes the same function faster, the way Qwenputer uses a custom executor for long runs:

| Unit | Dense | Fused |
|---|---|---|
| Memory | A float64 delta-rule state, one row per byte; one-hot keys and matrix products, O(memory) per access | Bytes in shared 4 KiB pages with copy-on-write (see below); indexing, O(1) per access |
| Bits of a value | 35 threshold layers | Shifts and masks |
| Byte to bits, 5-bit index | One-hot times a table | Index into the table |
| Division | 32 unrolled layers | Integer division, with RISC-V's rules for zero |

Tests check that the two agree unit by unit on random values, edge cases included, and over whole programs. Either way, the multiplier and divider only run when some guest in the batch needs them; otherwise their results would be routed nowhere, so skipping them changes nothing.

For GPUs the fused executor has a **static** mode, which always runs every unit, and **flat** memory, a row of bytes per guest on the device, with gated-off writes sent to a scratch byte past the end. A step then has no control flow that depends on data, so it can be compiled and captured whole (see [On GPUs](#on-gpus)). A test checks that static mode with flat memory matches the paged executor over a batch of programs.

## Verification

- **Unit tests** ([`neural/tests`](../neural/tests/)): instruction semantics including every M-extension edge case, exits and their details, faults that change nothing, snapshots, dense against fused, and batching.
- **Differential tests** ([`crates/shem/tests/neural.rs`](../crates/shem/tests/neural.rs)), the neural vCPU against the reference:
  - all 60 official riscv-tests (rv32ui, rv32um, rv32ua) run through containers on both, with the same exit status, the same instruction count, and the same final pc, registers and memory;
  - random straight-line programs over every register and immediate ALU instruction;
  - a snapshot taken midway on the reference vCPU, resumed on the neural one;
  - a container that forks, pipes a byte from child to parent and reaps the child, which exercises cloning the neural vCPU;
  - VM mode, with RAM at `0x80000000`: machine-mode programs that take traps and timer interrupts and idle in `wfi`, with the same console output, the same CSRs and timer, and the same final state;
  - **Linux**: a no-MMU Linux 6.8, booted to its prompt on the reference vCPU, then thawed on both vCPUs, where it reads `echo $((6 * 7))` from the console, runs it and prints the next prompt. That is about 180,000 instructions of kernel and BusyBox, through timer interrupts, CSR emulation, device accesses and atomics, and every byte of the 64 MiB of RAM ends the same. It takes the neural vCPU about four minutes.

`just neural-tests` runs them all, and CI runs them on every push.

## Density

Two things let one executor hold and run many guests.

**Paged memory with copy-on-write** ([`memory.py`](../neural/src/shem_neural/memory.py)). The fused executor keeps memory as bytes in 4 KiB pages from a pool shared by every vCPU of the same memory size, with a page table per vCPU. A gated write with β = 1 is a store and β = 0 is nothing, exactly as in the delta rule, and tests check paged and delta-rule memory against each other on random traffic. Pages make guests cheap:

- A fresh guest maps the zero page everywhere, so memory it never touches costs nothing.
- `CLONE`, which is how `fork` reaches the model, copies the page table; a shared page is copied only when one of its holders first writes to it.
- Whole pages the host writes, as `RESTORE` does with a snapshot, are interned by content. Containers resumed from the same warm image therefore share every page they have not written, as Linux's KSM would arrange.

A test clones a vCPU 20 times and restores its snapshot into 20 more, and the pool holds no extra pages.

**Continuous batching** ([`server.py`](../neural/src/shem_neural/server.py)). One executor thread steps every vCPU with a `RUN` in flight in the same forward pass. Their states stack cheaply, since registers and page tables are small and pages stay in the pool. A guest that exits or runs out of budget stops committing and gets its result; when another `RUN` arrives, the batch is rebuilt to include it. So independent containers, for example several `shem run --vcpu neural` processes, share forward passes without knowing about each other. A test runs six clients concurrently and checks every result, and that they shared a batch.

## Speed

On one CPU core, the fused executor runs about 900 instructions per second for one guest. That is the same order as Qwenputer's published speed on a GPU. A forward pass is a few hundred small tensor operations, so the time goes to per-operation overhead rather than arithmetic, and a batch of guests costs little more per step than one. [`bench.py`](../neural/bench.py), with every guest running a different loop:

| Guests | Instructions/s, total | Per guest | Memory pages |
|---|---|---|---|
| 1 | 881 | 881 | 2 |
| 8 | 5,725 | 716 | 9 |
| 32 | 18,384 | 575 | 33 |
| 64 | 27,985 | 437 | 65 |
| 128 | 29,318 | 229 | 129 |

Throughput grows almost linearly to 32 guests and keeps growing to 64, where one core runs 32 times as many instructions per second as with a single guest. This is design rule 2 measured: diverging guests do not slow each other down. The pages column counts the pool, the zero page included: each guest here holds one private page, 4 KiB, where a float64 delta-rule state for 1 MiB of memory would take 8 MiB.

## The protocol

The host talks to the server ([`server.py`](../neural/src/shem_neural/server.py), [`neural.rs`](../crates/shem/src/neural.rs)) over a Unix socket. Every message is a frame: a little-endian u32 length, then the payload. A request is an op byte, the vCPU id (except for `CREATE`) and the arguments; a response is a status byte, 0 for OK, 1 for an error with a message, 2 for a memory fault, and then the result. All integers are little-endian.

| Op | Arguments | Result |
|---|---|---|
| 1 `CREATE` | u32 memory size, optionally u32 base address (default 0) | u32 id |
| 2 `DROP` | | |
| 3 `CLONE` | | u32 id of the copy |
| 4 `RUN` | u64 budget | u8 exit reason, u32 detail, u32 pc, u64 instructions |
| 5 `GET_REG` | u8 index | u32 |
| 6 `SET_REG` | u8 index, u32 value | |
| 7 `GET_PC` | | u32 |
| 8 `SET_PC` | u32 | |
| 9 `READ_MEM` | u32 address, u32 length | bytes |
| 10 `WRITE_MEM` | u32 address, bytes | |
| 11 `RESET` | | |
| 12 `SNAPSHOT` | | canonical machine state |
| 13 `RESTORE` | canonical machine state | |

Exit reasons are 0 budget, 1 ecall, 2 ebreak, 3 illegal instruction (detail: the word), and 4 memory fault (detail: the address). Snapshots use the canonical format of `MachineState` in [`vcpu.rs`](../crates/shem/src/vcpu.rs), so a snapshot from either vCPU restores on the other.

## On GPUs

The network runs on whatever device its weights are on. [`gpu.py`](../neural/src/shem_neural/gpu.py) steps a batch of guests in place with the static executor and flat memory, in four modes: eager, compiled with `torch.compile` (which generates Triton kernels), eager in a CUDA graph that captures 64 steps so they launch as one, and both.

[`jobs/gpu.py`](../neural/jobs/gpu.py) is a uv script that runs on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) (`just gpu-job l4x1`). It downloads shem at a given commit and does two things:

1. **A differential test.** 1,024 guests run random RV32IM programs from random register files for 512 steps (about 100,000 instructions, ending in every kind of exit) on the CPU's paged executor, and then in each GPU mode. Every register, byte of memory, pc, instruction count, exit and exit detail must match. They do, in all four modes, on an L4 and an A100: GPU execution is bit-exact. Nothing relies on how the GPU rounds, because every intermediate value is an integer below 2^53, which float64 holds exactly, fused multiply-adds included.
2. **Throughput.** Every guest runs a loop of multiplies, divides, loads, stores, shifts and branches, at batch sizes from 1 to 32,768, with 64 KiB of memory each.

Instructions per second, with torch 2.15 and CUDA 13.2 (the CPU row is the paged executor on one core, from [Speed](#speed)):

| Guests | CPU | L4, best mode | A100, eager | A100, compiled + graph | A100, eager + graph |
|---|---|---|---|---|---|
| 1 | 881 | 4,818 | 192 | **12,859** | 2,086 |
| 64 | 27,985 | 278,995 | 12,187 | **577,138** | 121,028 |
| 1,024 | | 891,211 | 189,646 | **2,821,098** | 1,399,234 |
| 4,096 | | 998,556 | 760,787 | 3,549,051 | **3,664,716** |
| 16,384 | | 945,402 | 3,026,696 | 3,986,316 | **6,556,971** |
| 32,768 | | 968,596 | 6,118,754 | 3,166,873 | **7,434,596** |

What the numbers say:

- **Small batches are bound by launches.** Eager mode runs about 190 steps per second whatever the batch: each step is a few hundred kernel launches. Compiling fuses them into a few kernels and the graph launches 64 steps at once, which together make one guest 67 times faster than eager, and 15 times faster than one CPU core.
- **Large batches are bound by float64.** The A100 reaches 7.4 million instructions per second with 32,768 guests, 250 times a CPU core's best. The L4 levels off near a million, because it runs float64 at 1/64 of its float32 rate. At large batches the eager path in a graph beats the compiled one: cuBLAS does the batched matrix products better than the kernels Inductor generates for them.
- **The state is mostly bits.** Registers are 32 bits of float64 each, 8 KiB per guest, read and rewritten every step. Bits are exact in any float type, so keeping them in float32 or bfloat16, and only values in float64, would cut both the traffic and the float64 work.

## Running it

```sh
just neural-server                  # serves on /tmp/shem-neural.sock
shem run --vcpu neural --neural-socket /tmp/shem-neural.sock --mem 1M program
```
