# VM mode

`shem boot` runs a whole guest operating system: a no-MMU Linux kernel, with its console on stdio.

```sh
just linux-image                 # a prebuilt Linux 6.8 for RV32 without an MMU, with a BusyBox initramfs
shem boot target/linux/Image     # boots to a root shell in about two seconds
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

In container mode the host is the guest's kernel. In VM mode it is the hardware around the guest's own kernel ([`vm.rs`](../crates/shem/src/vm.rs)). The vCPU is the same RV32IM vCPU either way; everything else is the host's, done when the vCPU exits, which is design rule 1 in [architecture.md](architecture.md) taken to its end:

| The guest does | The vCPU | The host |
|---|---|---|
| A CSR instruction, `mret`, `wfi` | Exits: illegal instruction | Carries it out against machine-mode CSRs it keeps: `mstatus`, `mie`, `mip`, `mtvec`, `mscratch`, `mepc`, `mcause`, `mtval`, and the counters |
| `ecall`, `ebreak`, an illegal instruction, a bad access | Exits | Delivers a trap, as hardware does: `mepc`, `mcause` and `mtval` set, interrupts disabled, the previous privilege saved, and a jump to `mtvec` |
| Atomics | Exits: illegal instruction | Emulates them, as in container mode |
| A load or store outside RAM | Exits: memory fault | Decodes the instruction and emulates the device, as KVM leaves MMIO to userspace |
| Waits for its timer | | Raises the timer interrupt between slices, and on `wfi` skips straight to it |

Time is virtual: the timer ticks at 1 MHz, one tick per 100 instructions, as if the CPU ran at 100 MHz. Runs are therefore repeatable, and idle time costs nothing.

## The machine

It is the machine [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima) defines, so its Linux images boot unchanged:

| Address | Device |
|---|---|
| `0x80000000` | RAM (64 MiB by default), with the kernel at its start and the device tree at its top |
| `0x10000000` | 16550 UART: the console |
| `0x11000000` | CLINT: the timer (`mtime`, `mtimecmp`) |
| `0x11100000` | syscon: writing `0x5555` powers off, `0x7777` reboots |

The host generates the device tree ([`fdt.rs`](../crates/shem/src/fdt.rs)) for the memory size it is given, and boots the kernel by the RISC-V boot protocol: in machine mode, at the start of RAM, with the hart id in `a0` and the device tree's address in `a1`.

## Tests

- [`tests/vm.rs`](../crates/shem/tests/vm.rs) runs small machine-mode programs that read and write CSRs, take `ecall`, illegal-instruction and access-fault traps, return with `mret`, take timer interrupts, skip idle time with `wfi`, print through the UART and power off.
- `just vm-tests` downloads the Linux image, pinned by commit and checksum, boots it, waits for the shell prompt, types `uname -sm` and `echo $((6 * 7))`, checks the answers, and types `poweroff`. CI runs it on every push.

## Not yet

- **VM mode on the neural vCPU.** The kernel expects RAM at `0x80000000`; the reference vCPU has a base address for that, and the neural vCPU does not yet.
- **Booted snapshots.** Freezing a booted machine, as `WARM` does for containers, so an image can ship a kernel that has already booted.
- **Interrupt-driven devices.** The UART is polled; there is no interrupt controller, and no block device or network.
