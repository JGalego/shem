# The neural vCPU

The neural vCPU is an RV32IM CPU compiled into network weights. One forward pass executes one instruction per guest in a batch. It implements the same [`Vcpu` interface](../crates/shem/src/vcpu.rs) as the reference interpreter, accessed over a Unix socket.

Like [Qwenputer](https://divagr.com/blog/qwenputer), it uses compiled CPU layers and delta-rule memory. Unlike Qwenputer, it is a standalone network, not part of a pretrained language model.

## Running it

With uv installed, start the server from the `shem-neural` release that matches shem:

```sh
uvx --from "shem-neural==$(shem --version | cut -d' ' -f2)" shem-neural-server --socket /tmp/shem-neural.sock
```

From a repository checkout, `just neural-server` runs the checkout's own.

In another terminal, compile and run a guest (compilation needs Docker):

```sh
just cc -O2 -o hello guests/hello.c
shem run --vcpu neural --neural-socket /tmp/shem-neural.sock hello
```

The server uses the CPU. GPU execution is currently available only through the [benchmark harness](#on-gpus).

## Compiled, not trained

[`isa.py`](../neural/src/shem_neural/isa.py) defines 48 RV32IM instructions: decode patterns, operand formats, result routing, next pc and exits. [`compile.py`](../neural/src/shem_neural/compile.py) turns that table into tensors:

| Tensor | Shape | Role |
|---|---|---|
| `decode_w`, `decode_c` | 48 × 32, 48 | A layer that counts, for each instruction, how many masked bits of the word differ from its pattern. `relu(1 − count)` is a one-hot over instructions, and all zeros for an illegal one. |
| `index_w`, `index_c` | 32 × 5, 32 | The same trick for 5-bit register numbers and shift amounts |
| `imm_bits` | 5 × 32 × 32 | Selection matrices from instruction bits to immediate bits, one per format, sign extension included |
| `format_route`, `result_route`, `next_pc_route`, `branch_route`, `exit_route` | 48 × n | Routing tables: which format, result, next pc, comparison and exit each instruction uses |
| `uses_imm`, `load_size`, `store_size` | 48 | Per-instruction flags and sizes |
| `sll`, `srl` | 32 × 32 × 32 | One shift matrix per shift amount; the shifter is attention over them |
| `sra_fill` | 32 × 32 | Sign-bit fill for each arithmetic-right-shift amount |
| `byte_bits`, `powers` | 256 × 8, 32 | Byte-to-bits lookup table, and the weights that turn bits into a value |

Weights are saved as safetensors for inclusion in [images](shemfile.md).

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

Routing is per guest, so one batch can run unrelated programs.

## The operations it uses

The dense executor uses products with compiled weights (linear layers and one-hot attention), element-wise sums and products (gating), and clamps, which are piecewise-linear like ReLU. State changes only through gated delta-rule writes.

Integer intermediates are held exactly in float64. Lower-precision execution would need smaller arithmetic units, such as bytes rather than words.

## Two executors

The **dense** executor defines the network. The **fused** executor computes the same function with cheaper operations:

| Unit | Dense | Fused |
|---|---|---|
| Memory | A float64 delta-rule state, one row per byte; one-hot keys and matrix products, O(memory) per access | Bytes in shared 4 KiB pages with copy-on-write (see below); indexing, O(1) per access |
| Register file | One-hot reads, and delta-rule writes over the whole matrix | Gathers of the two operand rows, and the same write to the one row it can change |
| Shifter | Attention over the shift matrices | Integer shifts |
| Bits of a value | 35 threshold layers | Shifts and masks |
| Byte to bits, 5-bit index | One-hot times a table | Index into the table |
| Division | 32 unrolled layers | Integer division, with RISC-V's rules for zero |

Both skip multiplication and division when no guest needs them. Tests compare unit results, edge cases and whole programs.

For GPUs, **static** mode runs every unit and uses **flat** memory: one byte array per guest, with disabled writes sent to a scratch byte. This removes data-dependent control flow for compilation and CUDA graphs ([On GPUs](#on-gpus)).

## Verification

- [Unit tests](../neural/tests/): instruction semantics, M-extension edge cases, exits, non-mutating faults, snapshots, executor equivalence and batching.
- [Differential tests](../crates/shem/tests/neural.rs): all 60 official riscv-tests, random ALU programs, cross-vCPU snapshot resume, fork/pipe/wait, VM traps and timers.
- Linux resume: boot no-MMU Linux 6.8 on the reference vCPU, resume on both, and run `echo $((6 * 7))`. About 180,000 instructions; final registers, host device state and all 64 MiB of RAM must agree. Neural execution takes about four minutes.

Run with `just neural-tests`.

## Density

**Paged memory with copy-on-write** ([`memory.py`](../neural/src/shem_neural/memory.py)). The fused executor shares a pool of 4 KiB byte pages across vCPUs of the same memory size:

- A fresh guest maps the zero page everywhere, so memory it never touches costs nothing.
- `CLONE` copies the page table; writes copy shared pages on demand.
- `RESTORE` interns whole pages by content, so guests from the same warm image share unchanged pages.

**Continuous batching** ([`server.py`](../neural/src/shem_neural/server.py)). One executor thread groups pending `RUN` requests by CPU weights and memory size. It stacks registers and page tables, steps each group together, and returns guests that exit or exhaust their budget. New runs join between steps.

## Speed

Recorded CPU results from [`bench.py`](../neural/bench.py), using one core and a sum loop with a different bound for each guest:

| Guests | Instructions/s, total | Per guest | Memory pages |
|---|---|---|---|
| 1 | 881 | 881 | 2 |
| 8 | 5,725 | 716 | 9 |
| 32 | 18,384 | 575 | 33 |
| 64 | 27,985 | 437 | 65 |
| 128 | 29,318 | 229 | 129 |

At 64 guests, aggregate throughput is about 32 times the single-guest result, but per-guest throughput has fallen from 881 to 437 instructions/s. Batching improves throughput, not individual guest latency. The pages column includes the shared zero page and one private 4 KiB page per guest; a dense float64 state for 1 MiB of guest RAM takes 8 MiB.

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

[`jobs/gpu.py`](../neural/jobs/gpu.py) runs on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs):

```sh
just gpu-job l4x1
```

It compares 1,024 random-program guests against the CPU executor for 512 steps in each GPU mode, then benchmarks batches of 1–32,768 guests with 64 KiB each. Registers, memory, pc, instruction count and exits match bit-for-bit on L4 and A100. Integer intermediates stay below 2^53 and are exactly representable in float64.

Instructions per second, with torch 2.15 and CUDA 13.2 (the CPU row is the paged executor on one core, from [Speed](#speed)):

| Guests | CPU | L4, best mode | A100, eager | A100, compiled + graph | A100, eager + graph |
|---|---|---|---|---|---|
| 1 | 881 | 4,818 | 192 | **12,859** | 2,086 |
| 64 | 27,985 | 278,995 | 12,187 | **577,138** | 121,028 |
| 1,024 | | 891,211 | 189,646 | **2,821,098** | 1,399,234 |
| 4,096 | | 998,556 | 760,787 | 3,549,051 | **3,664,716** |
| 16,384 | | 945,402 | 3,026,696 | 3,986,316 | **6,556,971** |
| 32,768 | | 968,596 | 6,118,754 | 3,166,873 | **7,434,596** |

- Small batches are launch-bound; compilation and graphs reduce launches.
- Large batches are float64-bound. The A100 reaches 7.4 million instructions/s; the L4 levels off near one million.
- Register bits currently occupy 8 KiB per guest in float64. Lower-precision storage for bits could reduce memory traffic while retaining float64 arithmetic values.
