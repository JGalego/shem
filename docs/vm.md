# VM mode

`shem boot` runs a whole guest operating system: a no-MMU Linux kernel, with its console on stdio. A VM image can ship a kernel that has already booted, so `shem run` starts at its shell prompt.

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

In container mode the host is the guest's kernel. In VM mode it is the hardware around the guest's own kernel ([`vm.rs`](../crates/shem/src/vm.rs)). The vCPU is the same RV32IM vCPU either way; everything else is the host's, done when the vCPU exits, which is design rule 1 in [architecture.md](architecture.md) taken to its end:

| The guest does | The vCPU | The host |
|---|---|---|
| A CSR instruction, `mret`, `wfi` | Exits: illegal instruction | Carries it out against machine-mode CSRs it keeps: `mstatus`, `mie`, `mip`, `mtvec`, `mscratch`, `mepc`, `mcause`, `mtval`, and the counters |
| `ecall`, `ebreak`, an illegal instruction, a bad access | Exits | Delivers a trap, as hardware does: `mepc`, `mcause` and `mtval` set, interrupts disabled, the previous privilege saved, and a jump to `mtvec` |
| Atomics | Exits: illegal instruction | Emulates them, as in container mode |
| A load or store outside RAM | Exits: memory fault | Decodes the instruction and emulates the device, as KVM leaves MMIO to userspace |
| Waits for its timer | | Raises the timer interrupt between slices, and on `wfi` skips straight to it |

Time is virtual: the timer ticks at 1 MHz, one tick per 100 instructions, as if the CPU ran at 100 MHz. Runs are therefore repeatable, and idle time costs nothing. The host stops the vCPU exactly where `mtime` reaches `mtimecmp`, so the interrupt arrives at the same instruction however a run is split into slices, or frozen and thawed.

## The machine

It is the machine [mini-rv32ima](https://github.com/cnlohr/mini-rv32ima) defines, so its Linux images boot unchanged:

| Address | Device |
|---|---|
| `0x80000000` | RAM (64 MiB by default), with the kernel at its start and the device tree at its top |
| `0x10000000` | 16550 UART: the console |
| `0x11000000` | CLINT: the timer (`mtime`, `mtimecmp`) |
| `0x11100000` | syscon: writing `0x5555` powers off, `0x7777` reboots |

The host generates the device tree ([`fdt.rs`](../crates/shem/src/fdt.rs)) for the memory size it is given, and boots the kernel by the RISC-V boot protocol: in machine mode, at the start of RAM, with the hart id in `a0` and the device tree's address in `a1`.

## Booted snapshots

A machine freezes between instructions into the vCPU's canonical machine state, the same format containers use, and `vm.json`, the host's half of the hardware: the CSRs, the privilege level, `mtimecmp`, the time skipped while idle, the LR/SC reservation and the instruction count. Thawing it onto a fresh vCPU continues exactly where it stopped: a test freezes a program mid-run with its timer armed, thaws it, and checks that the output, the instruction count and the final state are those of an uninterrupted run.

A **VM image** names a kernel in its filesystem with `KERNEL`, and `WARM` boots it at build time ([shemfile.md](shemfile.md#vm-images)):

```dockerfile
FROM scratch
COPY Image /boot/Image
KERNEL /boot/Image
WARM --until=output:"~ # "
```

`--until=output:TEXT` stops once the console's output ends with `TEXT`, here BusyBox's prompt; `--until=steps:N` also works. The image then holds the booted machine, 64 MiB of RAM included, and building it twice gives the same digest. `shem run` resumes it with the console on stdio; with no `WARM`, it boots the kernel. [`examples/linux`](../examples/linux/Shemfile) fetches the kernel in a Docker stage, pinned by checksum.

## Tests

- [`tests/vm.rs`](../crates/shem/tests/vm.rs) runs small machine-mode programs that read and write CSRs, take `ecall`, illegal-instruction and access-fault traps, return with `mret`, take timer interrupts, skip idle time with `wfi`, print through the UART and power off.
- They also freeze and thaw a machine mid-run, pause on console output, and build a VM image from a small kernel, cold and warm.
- `just vm-tests` downloads the Linux image, pinned by commit and checksum, boots it, waits for the shell prompt, types `uname -sm` and `echo $((6 * 7))`, checks the answers, and types `poweroff`. It also freezes Linux at its prompt and resumes it on a fresh vCPU, and builds a booted Linux image twice, checking that the digests match and that the image answers at its prompt. CI runs it on every push.

## On the neural vCPU

`--vcpu neural` works for `shem boot` and for VM images, as it does for containers. The kernel expects RAM at `0x80000000`, so the neural vCPU takes a base address, and the network computes every fetch, load and store as an offset from it ([neural.md](neural.md#one-forward-pass)). Booting takes about 350 million instructions, which is days at a thousand instructions per second, so the practical way is a booted image: `WARM` boots the kernel on the reference vCPU, and the neural vCPU resumes at the prompt. A test does exactly that and runs a shell command on both vCPUs, about 180,000 instructions, which end in identical states.

```sh
just neural-server &
echo 'echo $((6 * 7))' | shem run --vcpu neural --neural-socket /tmp/shem-neural.sock linux   # minutes, not seconds
```

## Not yet
- **Interrupt-driven devices.** The UART is polled; there is no interrupt controller, and no block device or network.
