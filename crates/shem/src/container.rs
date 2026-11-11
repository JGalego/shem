//! Container mode: a set of guest processes, each on its own vCPU, with the host
//! as their kernel.
//!
//! The scheduler runs each runnable process for a slice of instructions in
//! turn, which is all the preemption a guest needs (design rule 3). A process
//! whose system call cannot finish yet, such as a read from an empty pipe, is
//! parked with its pc still on the `ecall`, and the call is retried on its next
//! turn. `fork` copies the vCPU (design rule 4). The container ends when its
//! first process, init, exits.

use std::collections::BTreeMap;
use std::path::Path;

use thiserror::Error;

use crate::elf::{self, ElfError};
use crate::emulate::{Emulation, Emulator};
use crate::linux::{
    Action, Delivery, Exec, FreezeError, FrozenLinux, FrozenProcess, Linux, ProcEntry, Process,
    Report, Stdio, Termination, Wait,
};
use crate::vcpu::{A0, A7, Exit, ExitReason, MachineState, MemoryFault, SP, SnapshotError, Vcpu};
use serde::{Deserialize, Serialize};

/// Memory kept free for the stack at the top of guest memory, above the heap and mappings.
pub const STACK_SIZE: u32 = 1 << 20;
/// Instructions per turn. Each slice boundary is a preemption point.
pub const SLICE: u64 = 1 << 16;
/// The pid of the container's first process.
pub const INIT: u32 = 1;
/// Most processes a container holds at once, zombies included.
const MAX_PROCESSES: usize = 256;

const SIGILL: u32 = 4;
const SIGTRAP: u32 = 5;
const SIGSEGV: u32 = 11;
const SIGCHLD: u32 = 17;
const CLD_EXITED: u32 = 1;
const CLD_KILLED: u32 = 2;
const EPERM: u32 = 1;
const ESRCH: u32 = 3;
const ECHILD: u32 = 10;
const EAGAIN: u32 = 11;
const EFAULT: u32 = 14;

#[derive(Debug, Error)]
pub enum StartError {
    #[error(transparent)]
    Elf(#[from] ElfError),
    #[error("arguments and environment do not fit on the stack: {0}")]
    Stack(#[from] MemoryFault),
}

/// A signal the host's terminal sent, such as SIGINT for Ctrl-C, for the
/// container to deliver to the guest's foreground process group: the host
/// catches it and stores its number here, so it does not kill the host.
pub static TERMINAL_SIGNAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Where `run_until` should stop before init exits, as `WARM --until` asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pause {
    /// Just before init's first system call with this number is serviced.
    Syscall(u32),
    /// Once this many instructions have retired.
    Steps(u64),
    /// When init executes `ebreak`; it resumes after the `ebreak`.
    Ebreak,
}

/// How `run_until` ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Exited(u8),
    Paused,
}

/// A container frozen for a warm image: init's machine state on the vCPU, and
/// the host-side state that goes with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frozen {
    pub machine: MachineState,
    pub host: FrozenHost,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenHost {
    pub steps: u64,
    pub linux: FrozenLinux,
    pub process: FrozenProcess,
    pub reservation: Option<u32>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RunError {
    /// Init stopped on something the host cannot service.
    #[error("guest stopped: {:?} at pc {:#010x}", .0.reason, .0.pc)]
    Fault(Exit),
    #[error("guest did not exit within {0} instructions")]
    StepLimit(u64),
    #[error("every guest process is blocked waiting for another")]
    Deadlock,
}

/// One guest process: its kernel state, its vCPU, and where it is in its life.
pub struct Task<V> {
    pub process: Process,
    pub vcpu: V,
    emulator: Emulator,
    pub state: State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Runnable,
    /// Parked on a system call that will be retried.
    Blocked,
    /// Exited, and not yet reaped by its parent.
    Zombie(Termination),
}

pub struct Container<V> {
    pub linux: Linux,
    tasks: BTreeMap<u32, Task<V>>,
    next_pid: u32,
    /// Instructions retired so far, across all processes.
    pub steps: u64,
    /// Processes other than init that a fault killed, for diagnostics.
    pub faults: Vec<(u32, Exit)>,
    /// Print each system call and its outcome to stderr, like strace.
    pub trace: bool,
    pause: Option<Pause>,
    paused: bool,
}

impl<V: Vcpu + Clone> Container<V> {
    /// Start a container whose init process runs the static executable `elf`.
    pub fn from_elf(
        vcpu: V,
        elf: &[u8],
        argv: &[String],
        envp: &[String],
        stdio: Stdio,
    ) -> Result<Self, StartError> {
        let mut task = Task {
            process: Process::new(INIT, stdio, 0, 0),
            vcpu,
            emulator: Emulator::default(),
            state: State::Runnable,
        };
        let exe = argv.first().map_or(&b""[..], |arg| arg.as_bytes());
        load(&mut task, elf, exe, argv, envp)?;
        Ok(Self {
            linux: Linux::new(),
            tasks: BTreeMap::from([(INIT, task)]),
            next_pid: INIT + 1,
            steps: 0,
            faults: Vec::new(),
            trace: false,
            pause: None,
            paused: false,
        })
    }

    /// Resume a frozen container: init picks up exactly where it was frozen,
    /// with its standard streams on `stdio`. The root filesystem is set
    /// separately, from the image's layers.
    pub fn thaw(mut vcpu: V, frozen: &Frozen, stdio: Stdio) -> Result<Self, SnapshotError> {
        vcpu.restore(&frozen.machine)?;
        let task = Task {
            process: Process::thaw(&frozen.host.process, stdio),
            vcpu,
            emulator: Emulator::with_reservation(frozen.host.reservation),
            state: State::Runnable,
        };
        Ok(Self {
            linux: Linux::thaw(&frozen.host.linux),
            tasks: BTreeMap::from([(INIT, task)]),
            next_pid: INIT + 1,
            steps: frozen.host.steps,
            faults: Vec::new(),
            trace: false,
            pause: None,
            paused: false,
        })
    }

    /// Freeze the container for a warm image. Only a lone init can be frozen,
    /// holding nothing but its standard streams.
    pub fn freeze(&self) -> Result<Frozen, FreezeError> {
        if self.tasks.len() != 1 {
            return Err(FreezeError::Processes(self.tasks.len()));
        }
        let init = &self.tasks[&INIT];
        Ok(Frozen {
            machine: init.vcpu.snapshot(),
            host: FrozenHost {
                steps: self.steps,
                linux: self.linux.freeze(),
                process: init.process.freeze()?,
                reservation: init.emulator.reservation(),
            },
        })
    }

    /// Set init's working directory, a directory in its root filesystem.
    pub fn set_cwd(&mut self, path: &str) -> Result<(), crate::linux::Errno> {
        self.tasks.get_mut(&INIT).unwrap().process.set_cwd(path)
    }

    /// Give the guest a filesystem: the host directory `root` becomes `/`.
    pub fn set_root(&mut self, root: &Path) -> std::io::Result<()> {
        self.tasks.get_mut(&INIT).unwrap().process.set_root(root)
    }

    /// The first process, which lives as long as the container.
    pub fn init(&self) -> &Task<V> {
        &self.tasks[&INIT]
    }

    pub fn task(&self, pid: u32) -> Option<&Task<V>> {
        self.tasks.get(&pid)
    }

    /// Run until init exits, and return its exit status: the exit code, or 128
    /// plus the signal that killed it.
    pub fn run(&mut self, max_steps: Option<u64>) -> Result<u8, RunError> {
        match self.run_until(max_steps, None)? {
            Stop::Exited(status) => Ok(status),
            Stop::Paused => unreachable!("no pause was asked for"),
        }
    }

    /// Run until init exits, or until `pause` holds.
    pub fn run_until(
        &mut self,
        max_steps: Option<u64>,
        pause: Option<Pause>,
    ) -> Result<Stop, RunError> {
        (self.pause, self.paused) = (pause, false);
        loop {
            // A signal typed at the terminal goes to its foreground process group
            let typed = TERMINAL_SIGNAL.swap(0, std::sync::atomic::Ordering::SeqCst);
            if typed != 0 {
                let group = self.linux.foreground as i32;
                self.kill(INIT, -group, typed);
            }
            let mut progress = false;
            let pids: Vec<u32> = self.tasks.keys().copied().collect();
            for pid in pids {
                progress |= self.turn(pid, max_steps)?;
                if self.paused {
                    return Ok(Stop::Paused);
                }
                if let State::Zombie(termination) = self.tasks[&INIT].state {
                    return Ok(Stop::Exited(termination.status()));
                }
            }
            if !progress {
                return Err(RunError::Deadlock);
            }
        }
    }

    /// Give `pid` one turn, and report whether it got anywhere.
    fn turn(&mut self, pid: u32, max_steps: Option<u64>) -> Result<bool, RunError> {
        let Some(task) = self.tasks.get_mut(&pid) else {
            return Ok(false);
        };
        if let State::Zombie(_) = task.state {
            return Ok(false);
        }
        // Pending signals come first: a handler runs, or a default action kills.
        let parked = task.state == State::Blocked;
        match task.process.signals.deliver(&mut task.vcpu, parked) {
            Delivery::Killed(termination) => {
                self.terminate(pid, termination);
                return Ok(true);
            }
            Delivery::Handler => task.state = State::Runnable,
            Delivery::Nothing if parked => return Ok(self.syscall(pid)),
            Delivery::Nothing => {}
        }
        let mut budget = match max_steps {
            None => SLICE,
            Some(max) if self.steps >= max => return Err(RunError::StepLimit(max)),
            Some(max) => SLICE.min(max - self.steps),
        };
        if let Some(Pause::Steps(at)) = self.pause {
            if self.steps >= at {
                self.paused = true;
                return Ok(false);
            }
            budget = budget.min(at - self.steps);
        }
        let exit = task.vcpu.run(budget);
        self.steps += exit.steps;
        task.process.cpu_ns += exit.steps;
        let init = pid == INIT;
        match exit.reason {
            ExitReason::Budget => {}
            // Pause with the pc still on the ecall, so that resuming services
            // the call with the input it gets then.
            ExitReason::Ecall if init && self.pause == Some(Pause::Syscall(task.vcpu.reg(A7))) => {
                self.paused = true;
            }
            ExitReason::Ecall => {
                self.syscall(pid);
            }
            ExitReason::Ebreak if init && self.pause == Some(Pause::Ebreak) => {
                task.vcpu.set_pc(exit.pc.wrapping_add(4));
                self.paused = true;
            }
            ExitReason::IllegalInstruction(ins) => {
                let task = self.tasks.get_mut(&pid).unwrap();
                match task.emulator.emulate(&mut task.vcpu, ins) {
                    Emulation::Done => {}
                    Emulation::Unsupported => self.fault(pid, exit)?,
                    Emulation::Fault(addr) => {
                        let reason = ExitReason::MemoryFault(addr);
                        self.fault(pid, Exit { reason, ..exit })?;
                    }
                }
            }
            _ => self.fault(pid, exit)?,
        }
        Ok(true)
    }

    /// Service the system call `pid` is stopped on. Returns false if it blocked.
    fn syscall(&mut self, pid: u32) -> bool {
        let nr = self.tasks[&pid].vcpu.reg(A7);
        if Linux::needs_processes(nr) {
            self.linux.processes = self.proc_entries();
        }
        let task = self.tasks.get_mut(&pid).unwrap();
        let call = [17, 10, 11, 12, 13].map(|reg| task.vcpu.reg(reg));
        let action = self
            .linux
            .handle(&mut task.process, &mut task.vcpu, self.steps);
        if self.trace {
            trace(pid, call, &action);
        }
        match action {
            Action::Return(value) => self.finish(pid, value),
            Action::Block => {
                task.state = State::Blocked;
                return false;
            }
            Action::Exit(termination) => self.terminate(pid, termination),
            Action::Fork { stack, parent_tid } => {
                let result = self.fork(pid, stack, parent_tid);
                self.finish(pid, result);
            }
            Action::Exec(exec) => self.exec(pid, exec),
            Action::Wait(wait) => return self.wait(pid, wait),
            Action::Kill {
                pid: target,
                signal,
            } => self.kill(pid, target, signal),
            Action::Resume => task_state(&mut self.tasks, pid, State::Runnable),
            Action::SetPgid { pid: child, pgid } => {
                let result = match self.tasks.get_mut(&child) {
                    Some(task) if task.process.ppid == pid => {
                        task.process.pgid = pgid;
                        0
                    }
                    Some(_) => EPERM.wrapping_neg(),
                    None => ESRCH.wrapping_neg(),
                };
                self.finish(pid, result);
            }
            Action::GetPgid { pid: target } => {
                let result = (self.tasks.get(&target))
                    .map_or(ESRCH.wrapping_neg(), |task| task.process.pgid);
                self.finish(pid, result);
            }
        }
        true
    }

    /// The processes as `/proc` shows them.
    fn proc_entries(&self) -> Vec<ProcEntry> {
        (self.tasks.values())
            .map(|task| {
                let state = match task.state {
                    State::Runnable => b'R',
                    State::Blocked => b'S',
                    State::Zombie(_) => b'Z',
                };
                task.process.proc_entry(state)
            })
            .collect()
    }

    /// Return `value` from the system call `pid` is stopped on.
    fn finish(&mut self, pid: u32, value: u32) {
        let task = self.tasks.get_mut(&pid).unwrap();
        task.vcpu.set_reg(A0, value);
        task.vcpu.set_pc(task.vcpu.pc().wrapping_add(4));
        task.state = State::Runnable;
    }

    fn fork(&mut self, parent: u32, stack: u32, parent_tid: u32) -> u32 {
        if self.tasks.len() >= MAX_PROCESSES {
            return EAGAIN.wrapping_neg();
        }
        let pid = self.next_pid;
        let task = self.tasks.get_mut(&parent).unwrap();
        if parent_tid != 0 && task.vcpu.write_mem(parent_tid, &pid.to_le_bytes()).is_err() {
            return EFAULT.wrapping_neg();
        }
        self.next_pid += 1;
        let mut child = Task {
            process: task.process.fork(pid),
            vcpu: task.vcpu.clone(),
            emulator: Emulator::default(),
            state: State::Runnable,
        };
        // The child returns 0 from the same `ecall`, on its own stack if given one.
        child.vcpu.set_reg(A0, 0);
        child.vcpu.set_pc(child.vcpu.pc().wrapping_add(4));
        if stack != 0 {
            child.vcpu.set_reg(SP, stack);
        }
        self.tasks.insert(pid, child);
        pid
    }

    /// Replace the image of `pid`. `Linux` has already checked that the program
    /// loads, so the only failure left is arguments that overflow the stack.
    /// That is past the point of no return, and Linux kills the process then too.
    fn exec(&mut self, pid: u32, exec: Exec) {
        let task = self.tasks.get_mut(&pid).unwrap();
        task.process.sync_shared(&task.vcpu, None);
        task.vcpu.reset();
        task.emulator = Emulator::default();
        task.state = State::Runnable;
        if load(task, &exec.elf, &exec.path, &exec.argv, &exec.envp).is_err() {
            self.terminate(pid, Termination::Killed(SIGSEGV));
        }
    }

    /// Reap a zombie child of `pid`, or block until there is one. Returns false
    /// if it blocked.
    fn wait(&mut self, pid: u32, wait: Wait) -> bool {
        let children: Vec<(u32, State)> = self
            .tasks
            .iter()
            .filter(|(child, task)| {
                task.process.ppid == pid && wait.pid.is_none_or(|wanted| wanted == **child)
            })
            .map(|(&child, task)| (child, task.state))
            .collect();
        if children.is_empty() {
            self.finish(pid, ECHILD.wrapping_neg());
            return true;
        }
        let zombie = children.iter().find_map(|&(child, state)| match state {
            State::Zombie(termination) => Some((child, termination)),
            _ => None,
        });
        let Some((child, termination)) = zombie else {
            if !wait.nohang {
                self.tasks.get_mut(&pid).unwrap().state = State::Blocked;
                return false;
            }
            let cleared = match wait.report {
                Report::Siginfo(info) => self.write(pid, info, &[0; 128]),
                Report::Status(_) => true,
            };
            self.finish(pid, if cleared { 0 } else { EFAULT.wrapping_neg() });
            return true;
        };
        let (written, value) = match wait.report {
            Report::Status(0) => (true, child),
            Report::Status(status) => {
                let word = termination.wait_status().to_le_bytes();
                (self.write(pid, status, &word), child)
            }
            Report::Siginfo(info) => {
                let (code, status) = match termination {
                    Termination::Exited(code) => (CLD_EXITED, code as u32),
                    Termination::Killed(signal) => (CLD_KILLED, signal),
                };
                let mut siginfo = [0u8; 128];
                for (offset, field) in [(0, SIGCHLD), (8, code), (12, child), (20, status)] {
                    siginfo[offset..offset + 4].copy_from_slice(&field.to_le_bytes());
                }
                (self.write(pid, info, &siginfo), 0)
            }
        };
        if !written {
            self.finish(pid, EFAULT.wrapping_neg());
            return true;
        }
        self.tasks.remove(&child);
        self.finish(pid, value);
        true
    }

    /// Send `signal` from `sender` to a pid, to a process group (0 for the
    /// sender's, -g for group g), or to every process but init (-1). Handlers
    /// never run, so a signal either kills its target or does nothing. Zombies
    /// count as targets, but are past killing.
    fn kill(&mut self, sender: u32, target: i32, signal: u32) {
        let group = |pgid: u32| -> Vec<u32> {
            let members = self
                .tasks
                .iter()
                .filter(|(_, task)| task.process.pgid == pgid);
            members.map(|(&pid, _)| pid).collect()
        };
        let targets: Vec<u32> = match target {
            1.. if self.tasks.contains_key(&(target as u32)) => vec![target as u32],
            1.. => Vec::new(),
            0 => group(self.tasks[&sender].process.pgid),
            -1 => self
                .tasks
                .keys()
                .copied()
                .filter(|&pid| pid != INIT)
                .collect(),
            _ => group(target.unsigned_abs()),
        };
        if targets.is_empty() {
            self.finish(sender, ESRCH.wrapping_neg());
            return;
        }
        for &pid in &targets {
            let task = self.tasks.get_mut(&pid).unwrap();
            if matches!(task.state, State::Zombie(_)) {
                continue;
            }
            if let Some(termination) = task.process.signals.raise(signal) {
                self.terminate(pid, termination);
            }
        }
        if !matches!(self.tasks[&sender].state, State::Zombie(_)) {
            self.finish(sender, 0);
        }
    }

    /// A process other than init that faults is killed, as Linux would with
    /// SIGSEGV or SIGILL. A fault in init stops the container with the details.
    fn fault(&mut self, pid: u32, exit: Exit) -> Result<(), RunError> {
        if pid == INIT {
            return Err(RunError::Fault(exit));
        }
        self.faults.push((pid, exit));
        let signal = match exit.reason {
            ExitReason::IllegalInstruction(_) => SIGILL,
            ExitReason::Ebreak => SIGTRAP,
            _ => SIGSEGV,
        };
        self.terminate(pid, Termination::Killed(signal));
        Ok(())
    }

    /// End `pid`: close its files, which closes pipe ends, and hand its children
    /// to init.
    fn terminate(&mut self, pid: u32, termination: Termination) {
        let task = self.tasks.get_mut(&pid).unwrap();
        task.state = State::Zombie(termination);
        task.process.sync_shared(&task.vcpu, None);
        task.process.release();
        let ppid = task.process.ppid;
        if let Some(parent) = self.tasks.get_mut(&ppid) {
            // SIGCHLD is ignored by default, so this only queues it for a handler.
            parent.process.signals.raise(SIGCHLD);
        }
        for task in self.tasks.values_mut() {
            if task.process.ppid == pid {
                task.process.ppid = INIT;
            }
        }
    }

    fn write(&mut self, pid: u32, addr: u32, bytes: &[u8]) -> bool {
        let task = self.tasks.get_mut(&pid).unwrap();
        task.vcpu.write_mem(addr, bytes).is_ok()
    }
}

/// Load a program into `task`: its segments, a fresh address space, and the
/// initial stack, ready to start at the entry point. `exe` is its guest path.
fn load<V: Vcpu>(
    task: &mut Task<V>,
    elf: &[u8],
    exe: &[u8],
    argv: &[impl AsRef<[u8]>],
    envp: &[impl AsRef<[u8]>],
) -> Result<(), StartError> {
    let image = elf::load(&mut task.vcpu, elf)?;
    let stack_top = task.vcpu.mem_size();
    let sp = elf::setup_stack(&mut task.vcpu, &image, stack_top, argv, envp)?;
    task.process
        .exec(image.brk, stack_top.saturating_sub(STACK_SIZE));
    task.process.set_command(argv, exe);
    task.vcpu.set_reg(SP, sp);
    task.vcpu.set_pc(image.entry);
    Ok(())
}

/// Print one system call the way `strace` would, more or less.
fn trace(pid: u32, [nr, args @ ..]: [u32; 5], action: &Action) {
    let outcome = match action {
        Action::Return(value) => (*value as i32).to_string(),
        Action::Exec(exec) => {
            let argv: Vec<_> = exec
                .argv
                .iter()
                .map(|a| String::from_utf8_lossy(a))
                .collect();
            format!("exec {argv:?}")
        }
        other => format!("{other:?}"),
    };
    let [a0, a1, a2, a3] = args;
    eprintln!("[{pid}] syscall {nr}({a0:#x}, {a1:#x}, {a2:#x}, {a3:#x}) -> {outcome}");
}

fn task_state<V>(tasks: &mut BTreeMap<u32, Task<V>>, pid: u32, state: State) {
    tasks.get_mut(&pid).unwrap().state = state;
}
