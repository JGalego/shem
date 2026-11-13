# VM mode

`shem boot` boots a no-MMU Linux kernel with its console on stdio. `WARM` saves a booted VM so `shem run` can resume at the shell prompt.

```sh
just linux-image                 # a prebuilt Linux 6.8 for RV32 without an MMU, with a BusyBox initramfs
shem boot target/linux/Image     # boots to a root shell in about two seconds

shem build -t linux examples/linux                  # boots it once, at build time, and freezes it at the prompt
printf 'uname -a\npoweroff\n' | shem run linux     # resumes at the prompt, in a fraction of a second
```

```
[    0.000000] Linux version 6.8.0-rc1mini-rv32ima ...
[    0.000000] Machine model: shem
...
Welcome to mini-rv32ima Linux
~ # uname -sm
Linux riscv32
~ # poweroff
```

## The host is the machine

In VM mode the host emulates hardware, not system calls ([`vm.rs`](../crates/shem/src/vm.rs)). Both modes use an RV32IM vCPU:

| The guest does | The vCPU | The host |
|---|---|---|
| A CSR instruction, `mret`, `wfi` | Exits: illegal instruction | Carries it out against machine-mode CSRs it keeps: `mstatus`, `mie`, `mip`, `mtvec`, `mscratch`, `mepc`, `mcause`, `mtval`, and the counters |
| `ecall`, `ebreak`, an illegal instruction, a bad access | Exits | Delivers a trap, as hardware does: `mepc`, `mcause` and `mtval` set, interrupts disabled, the previous privilege saved, and a jump to `mtvec` |
| Atomics | Exits: illegal instruction | Emulates them, as in container mode |
| A load or store outside RAM | Exits: memory fault | Decodes the instruction and emulates the device, as KVM leaves MMIO to userspace |
| Waits for its timer | | Raises the timer interrupt between slices, and on `wfi` skips straight to it |

The virtual timer ticks at 1 MHz, once per 100 instructions. Execution stops exactly when `mtime` reaches `mtimecmp`, independent of slice boundaries or snapshots. `wfi` skips idle time.

## The machine

The machine is compatible with [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima):

| Address | Device |
|---|---|
| `0x80000000` | RAM (64 MiB by default), with the kernel at its start and the device tree at its top |
| `0x10000000` | 16550 UART: the console |
| `0x11000000` | CLINT: the timer (`mtime`, `mtimecmp`) |
| `0x11100000` | syscon: writing `0x5555` powers off, `0x7777` reboots |

The host generates the device tree ([`fdt.rs`](../crates/shem/src/fdt.rs)) for the memory size it is given, and boots the kernel by the RISC-V boot protocol: in machine mode, at the start of RAM, with the hart id in `a0` and the device tree's address in `a1`.

## Booted snapshots

A snapshot contains canonical vCPU state and `vm.json`: CSRs, privilege level, timer state, skipped idle time, LR/SC reservation and instruction count. It can resume on either vCPU.

A VM image selects its kernel with `KERNEL` and optionally boots it at build time with `WARM`:

```dockerfile
FROM scratch
COPY Image /boot/Image
KERNEL /boot/Image
WARM --until=output:"~ # "
```

`--until=output:TEXT` stops when console output ends with `TEXT`; `--until=steps:N` stops after N instructions. The image includes RAM (64 MiB by default). Without `WARM`, `shem run` boots from scratch. The [Linux example](../examples/linux/Shemfile) downloads a checksum-pinned kernel.

## Tests

- [`tests/vm.rs`](../crates/shem/tests/vm.rs): CSRs, traps, timer interrupts, UART, poweroff, snapshots and VM image builds.
- `just vm-tests`: Linux boot, shell commands, snapshot resume and reproducible image digests.

## On the neural vCPU

Boot on the reference vCPU, then resume on the neural one. A cold boot takes about 350 million instructions: days at roughly 1,000 instructions/s. A shell command after resume takes about 180,000 instructions, or several minutes.

Start the server in a separate terminal:

```sh
just neural-server
```

Then resume the `linux` image built above:

```sh
printf 'echo $((6 * 7))\npoweroff\n' | \
  shem run --vcpu neural --neural-socket /tmp/shem-neural.sock linux
```

VM RAM starts at `0x80000000`; the network uses this base for [fetches, loads and stores](neural.md#one-forward-pass). `shem boot --vcpu neural` also works, but does not skip the cold boot.

## Not yet

- Interrupt controller, block devices and networking. The UART is polled.
