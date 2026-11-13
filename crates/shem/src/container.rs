//! Container scheduling and process lifecycle.
//!
//! Each process owns a vCPU; its threads share memory and swap register sets.
//! Runnable threads get instruction slices. A blocked system call keeps its pc
//! on the `ecall` for retry on the next turn. `fork` clones vCPU state, and
//! the container ends when init exits.

use std::cell::{Ref, RefCell, RefMut};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

use thiserror::Error;

use crate::elf::{self, ElfError};
use crate::emulate::{Emulation, Emulator};
use crate::linux::{
    Action, Delivery, Exec, FreezeError, FrozenLinux, FrozenProcess, Linux, ProcEntry, Process,
    Report, Stdio, Termination, Wait,
};
use crate::vcpu::{A0, A7, Exit, ExitReason, MachineState, MemoryFault, SP, SnapshotError, Vcpu};

/// The thread pointer register, which holds a thread's TLS.
const TP: usize = 4;
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

/// One guest thread: its process's kernel state and vCPU, which it shares
/// with the process's other threads, and where it is in its life.
pub struct Task<V> {
    /// The thread's id. A process's id is its first thread's.
    pub tid: u32,
    process: Rc<RefCell<Process>>,
    core: Rc<RefCell<Core<V>>>,
    emulator: Emulator,
    pub state: State,
}

impl<V: Vcpu> Task<V> {
    fn new(tid: u32, process: Process, vcpu: V, emulator: Emulator) -> Self {
        let core = Core {
            vcpu,
            loaded: tid,
            saved: HashMap::new(),
        };
        Self {
            tid,
            process: Rc::new(RefCell::new(process)),
            core: Rc::new(RefCell::new(core)),
            emulator,
            state: State::Runnable,
        }
    }

    /// The thread's process, shared with its other threads.
    pub fn process(&self) -> Ref<'_, Process> {
        self.process.borrow()
    }

    /// The process's vCPU, holding this thread's registers.
    pub fn vcpu(&self) -> RefMut<'_, V> {
        cpu(&self.core, self.tid)
    }

    /// Whether this is its process's first thread, which stands for the
    /// process: it is the one that `wait`, `kill` and `/proc` see.
    fn is_leader(&self) -> bool {
        self.tid == self.process.borrow().pid
    }

    fn same_process(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.process, &other.process)
    }
}

/// A thread's registers while another thread of its process has the vCPU.
struct Context {
    pc: u32,
    regs: [u32; 32],
}

/// A process's vCPU, which its threads take turns on: memory is the vCPU's,
/// so they share it, and each keeps its registers here while another runs.
struct Core<V> {
    vcpu: V,
    /// The thread whose registers the vCPU holds, or `NOBODY`.
    loaded: u32,
    saved: HashMap<u32, Context>,
}

/// No thread: the one whose registers the vCPU held has ended.
const NOBODY: u32 = u32::MAX;

/// The vCPU of `core` with `tid`'s registers in it, switching threads if
/// another's are there.
fn cpu<V: Vcpu>(core: &RefCell<Core<V>>, tid: u32) -> RefMut<'_, V> {
    RefMut::map(core.borrow_mut(), |core| {
        if core.loaded != tid {
            let vcpu = &mut core.vcpu;
            if core.loaded != NOBODY {
                let outgoing = Context {
                    pc: vcpu.pc(),
                    regs: std::array::from_fn(|i| vcpu.reg(i)),
                };
                core.saved.insert(core.loaded, outgoing);
            }
            if let Some(incoming) = core.saved.remove(&tid) {
                vcpu.set_pc(incoming.pc);
                for (i, &value) in incoming.regs.iter().enumerate().skip(1) {
                    vcpu.set_reg(i, value);
                }
            }
            core.loaded = tid;
        }
        &mut core.vcpu
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Runnable,
    /// Parked on a system call that will be retried.
    Blocked,
    /// A process's first thread, ended while its other threads go on.
    Gone,
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
        let mut task = Task::new(
            INIT,
            Process::new(INIT, stdio, 0, 0),
            vcpu,
            Emulator::default(),
        );
        let exe = argv.first().map_or(&b""[..], |arg| arg.as_bytes());
        load(&mut task, elf, exe, argv, envp)?;
        Ok(Self::with_init(task, Linux::new(), 0))
    }

    /// Resume a frozen container: init picks up exactly where it was frozen,
    /// with its standard streams on `stdio`. The root filesystem is set
    /// separately, from the image's layers.
    pub fn thaw(mut vcpu: V, frozen: &Frozen, stdio: Stdio) -> Result<Self, SnapshotError> {
        vcpu.restore(&frozen.machine)?;
        let process = Process::thaw(&frozen.host.process, stdio);
        let emulator = Emulator::with_reservation(frozen.host.reservation);
        let task = Task::new(INIT, process, vcpu, emulator);
        Ok(Self::with_init(
            task,
            Linux::thaw(&frozen.host.linux),
            frozen.host.steps,
        ))
    }

    fn with_init(task: Task<V>, linux: Linux, steps: u64) -> Self {
        Self {
            linux,
            tasks: BTreeMap::from([(INIT, task)]),
            next_pid: INIT + 1,
            steps,
            faults: Vec::new(),
            trace: false,
            pause: None,
            paused: false,
        }
    }

    /// Freeze the container for a warm image. Only a lone init can be frozen,
    /// with a single thread, holding nothing but its standard streams.
    pub fn freeze(&self) -> Result<Frozen, FreezeError> {
        if self.tasks.len() != 1 {
            return Err(FreezeError::Processes(self.tasks.len()));
        }
        let init = &self.tasks[&INIT];
        Ok(Frozen {
            machine: init.vcpu().snapshot(),
            host: FrozenHost {
                steps: self.steps,
                linux: self.linux.freeze(),
                process: init.process().freeze()?,
                reservation: init.emulator.reservation(),
            },
        })
    }

    /// Set init's working directory, a directory in its root filesystem.
    pub fn set_cwd(&mut self, path: &str) -> Result<(), crate::linux::Errno> {
        self.tasks[&INIT].process.borrow_mut().set_cwd(path)
    }

    /// Give the guest a filesystem: the host directory `root` becomes `/`.
    pub fn set_root(&mut self, root: &Path) -> std::io::Result<()> {
        self.tasks[&INIT].process.borrow_mut().set_root(root)
    }

    /// The first process, which lives as long as the container.
    pub fn init(&self) -> &Task<V> {
        &self.tasks[&INIT]
    }

    pub fn task(&self, tid: u32) -> Option<&Task<V>> {
        self.tasks.get(&tid)
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
            // A signal typed at the terminal goes to its foreground process
            // group, init's to begin with. The terminal sends it, so no
            // process's system call returns.
            let typed = TERMINAL_SIGNAL.swap(0, std::sync::atomic::Ordering::SeqCst);
            if typed != 0 {
                let foreground = self.group(self.linux.foreground);
                self.raise(&foreground, typed);
            }
            let mut progress = false;
            let tids: Vec<u32> = self.tasks.keys().copied().collect();
            for tid in tids {
                progress |= self.turn(tid, max_steps)?;
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

    /// Give thread `tid` one turn, and report whether it got anywhere.
    fn turn(&mut self, tid: u32, max_steps: Option<u64>) -> Result<bool, RunError> {
        let Some(task) = self.tasks.get_mut(&tid) else {
            return Ok(false);
        };
        if matches!(task.state, State::Zombie(_) | State::Gone) {
            return Ok(false);
        }
        // Pending signals come first: a handler runs, or a default action kills.
        let parked = task.state == State::Blocked;
        let delivery = {
            let mut vcpu = cpu(&task.core, tid);
            task.process
                .borrow_mut()
                .signals
                .deliver(&mut *vcpu, parked)
        };
        match delivery {
            Delivery::Killed(termination) => {
                self.terminate(tid, termination);
                return Ok(true);
            }
            Delivery::Handler => task.state = State::Runnable,
            Delivery::Nothing if parked => return Ok(self.syscall(tid)),
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
        let exit = cpu(&task.core, tid).run(budget);
        self.steps += exit.steps;
        task.process.borrow_mut().cpu_ns += exit.steps;
        let init = tid == INIT;
        match exit.reason {
            ExitReason::Budget => {}
            // Pause with the pc still on the ecall, so that resuming services
            // the call with the input it gets then.
            ExitReason::Ecall
                if init && self.pause == Some(Pause::Syscall(task.vcpu().reg(A7))) =>
            {
                self.paused = true;
            }
            ExitReason::Ecall => {
                self.syscall(tid);
            }
            ExitReason::Ebreak if init && self.pause == Some(Pause::Ebreak) => {
                task.vcpu().set_pc(exit.pc.wrapping_add(4));
                self.paused = true;
            }
            ExitReason::IllegalInstruction(ins) => {
                let task = self.tasks.get_mut(&tid).unwrap();
                let emulation = {
                    let mut vcpu = cpu(&task.core, tid);
                    task.emulator.emulate(&mut *vcpu, ins)
                };
                match emulation {
                    Emulation::Done => {}
                    Emulation::Unsupported => self.fault(tid, exit)?,
                    Emulation::Fault(addr) => {
                        let reason = ExitReason::MemoryFault(addr);
                        self.fault(tid, Exit { reason, ..exit })?;
                    }
                }
            }
            _ => self.fault(tid, exit)?,
        }
        Ok(true)
    }

    /// Service the system call thread `tid` is stopped on. Returns false if it
    /// blocked.
    fn syscall(&mut self, tid: u32) -> bool {
        let nr = self.tasks[&tid].vcpu().reg(A7);
        if Linux::needs_processes(nr) {
            self.linux.processes = self.proc_entries();
        }
        let task = self.tasks.get_mut(&tid).unwrap();
        let (call, action) = {
            let mut vcpu = cpu(&task.core, tid);
            let call = [17, 10, 11, 12, 13].map(|reg| vcpu.reg(reg));
            let mut process = task.process.borrow_mut();
            process.tid = tid;
            (
                call,
                self.linux.handle(&mut process, &mut *vcpu, self.steps),
            )
        };
        if self.trace {
            trace(tid, call, &action);
        }
        match action {
            Action::Return(value) => self.finish(tid, value),
            Action::Block => {
                task.state = State::Blocked;
                return false;
            }
            Action::Exit(termination) => self.terminate(tid, termination),
            Action::ExitThread(termination) => self.exit_thread(tid, termination),
            Action::Fork { stack, parent_tid } => {
                let result = self.fork(tid, stack, parent_tid);
                self.finish(tid, result);
            }
            Action::Thread {
                stack,
                tls,
                parent_tid,
                child_tid,
                clear_child_tid,
            } => {
                let result = self.thread(tid, stack, tls, (parent_tid, child_tid), clear_child_tid);
                self.finish(tid, result);
            }
            Action::Exec(exec) => self.exec(tid, exec),
            Action::Wait(wait) => return self.wait(tid, wait),
            Action::Kill {
                pid: target,
                signal,
            } => self.kill(tid, target, signal),
            Action::Resume => task_state(&mut self.tasks, tid, State::Runnable),
            Action::SetPgid { pid: child, pgid } => {
                let caller = self.pid_of(tid);
                let result = match self.leader(child) {
                    Some(task) if task.process().ppid == caller => {
                        task.process.borrow_mut().pgid = pgid;
                        0
                    }
                    Some(_) => EPERM.wrapping_neg(),
                    None => ESRCH.wrapping_neg(),
                };
                self.finish(tid, result);
            }
            Action::GetPgid { pid: target } => {
                let result =
                    (self.leader(target)).map_or(ESRCH.wrapping_neg(), |task| task.process().pgid);
                self.finish(tid, result);
            }
        }
        true
    }

    /// The process that thread `tid` belongs to.
    fn pid_of(&self, tid: u32) -> u32 {
        self.tasks[&tid].process().pid
    }

    /// The first thread of process `pid`, which stands for the process.
    fn leader(&self, pid: u32) -> Option<&Task<V>> {
        self.tasks.get(&pid).filter(|task| task.is_leader())
    }

    /// The other threads of `tid`'s process.
    fn siblings(&self, tid: u32) -> Vec<u32> {
        let task = &self.tasks[&tid];
        (self.tasks.iter())
            .filter(|&(&other, thread)| other != tid && thread.same_process(task))
            .map(|(&other, _)| other)
            .collect()
    }

    /// The processes as `/proc` shows them.
    fn proc_entries(&self) -> Vec<ProcEntry> {
        (self.tasks.values())
            .filter(|task| task.is_leader())
            .map(|task| {
                let state = match task.state {
                    State::Runnable => b'R',
                    State::Blocked | State::Gone => b'S',
                    State::Zombie(_) => b'Z',
                };
                task.process().proc_entry(state)
            })
            .collect()
    }

    /// Return `value` from the system call thread `tid` is stopped on.
    fn finish(&mut self, tid: u32, value: u32) {
        let task = self.tasks.get_mut(&tid).unwrap();
        {
            let mut vcpu = task.vcpu();
            vcpu.set_reg(A0, value);
            let pc = vcpu.pc();
            vcpu.set_pc(pc.wrapping_add(4));
        }
        task.state = State::Runnable;
    }

    fn fork(&mut self, parent: u32, stack: u32, parent_tid: u32) -> u32 {
        if self.tasks.len() >= MAX_PROCESSES {
            return EAGAIN.wrapping_neg();
        }
        let pid = self.next_pid;
        let task = &self.tasks[&parent];
        let (vcpu, process) = {
            let mut vcpu = task.vcpu();
            if parent_tid != 0 && vcpu.write_mem(parent_tid, &pid.to_le_bytes()).is_err() {
                return EFAULT.wrapping_neg();
            }
            // Only the calling thread is copied, as on Linux
            ((*vcpu).clone(), task.process().fork(pid))
        };
        self.next_pid += 1;
        let child = Task::new(pid, process, vcpu, Emulator::default());
        {
            // The child returns 0 from the same `ecall`, on its own stack if given one.
            let mut vcpu = child.vcpu();
            vcpu.set_reg(A0, 0);
            let pc = vcpu.pc();
            vcpu.set_pc(pc.wrapping_add(4));
            if stack != 0 {
                vcpu.set_reg(SP, stack);
            }
        }
        self.tasks.insert(pid, child);
        pid
    }

    /// Start a thread in `caller`'s process: the caller's registers, returning
    /// 0 from the `ecall` on its own stack, with its own thread pointer.
    fn thread(
        &mut self,
        caller: u32,
        stack: u32,
        tls: Option<u32>,
        (parent_tid, child_tid): (u32, u32),
        clear_child_tid: bool,
    ) -> u32 {
        if self.tasks.len() >= MAX_PROCESSES {
            return EAGAIN.wrapping_neg();
        }
        let tid = self.next_pid;
        let task = &self.tasks[&caller];
        let context = {
            let mut vcpu = task.vcpu();
            for addr in [parent_tid, child_tid] {
                if addr != 0 && vcpu.write_mem(addr, &tid.to_le_bytes()).is_err() {
                    return EFAULT.wrapping_neg();
                }
            }
            let mut regs: [u32; 32] = std::array::from_fn(|i| vcpu.reg(i));
            regs[A0] = 0;
            if stack != 0 {
                regs[SP] = stack;
            }
            if let Some(tls) = tls {
                regs[TP] = tls;
            }
            Context {
                pc: vcpu.pc().wrapping_add(4),
                regs,
            }
        };
        self.next_pid += 1;
        task.core.borrow_mut().saved.insert(tid, context);
        if clear_child_tid {
            task.process
                .borrow_mut()
                .clear_child_tid
                .insert(tid, child_tid);
        }
        let thread = Task {
            tid,
            process: task.process.clone(),
            core: task.core.clone(),
            emulator: Emulator::default(),
            state: State::Runnable,
        };
        self.tasks.insert(tid, thread);
        tid
    }

    /// End thread `tid`. Its id is cleared where it asked, which wakes threads
    /// that join it. The process ends with its last thread.
    fn exit_thread(&mut self, tid: u32, termination: Termination) {
        let live: Vec<u32> = (self.siblings(tid).into_iter())
            .filter(|other| !matches!(self.tasks[other].state, State::Zombie(_) | State::Gone))
            .collect();
        if live.is_empty() {
            self.terminate(tid, termination);
            return;
        }
        let task = &self.tasks[&tid];
        if let Some(addr) = task.process.borrow_mut().clear_child_tid.remove(&tid) {
            let _ = task.vcpu().write_mem(addr, &[0; 4]);
        }
        {
            let mut core = task.core.borrow_mut();
            core.saved.remove(&tid);
            if core.loaded == tid {
                core.loaded = NOBODY;
            }
        }
        if task.is_leader() {
            self.tasks.get_mut(&tid).unwrap().state = State::Gone;
        } else {
            self.tasks.remove(&tid);
        }
    }

    /// Replace the image of `tid`'s process. Its other threads end, and the
    /// caller goes on as the process's only thread, with the process's id.
    /// `Linux` has already checked that the program loads, so the only failure
    /// left is arguments that overflow the stack. That is past the point of no
    /// return, and Linux kills the process then too.
    fn exec(&mut self, tid: u32, exec: Exec) {
        let pid = self.pid_of(tid);
        {
            let task = &self.tasks[&tid];
            let vcpu = task.vcpu();
            task.process.borrow_mut().sync_shared(&*vcpu, None);
        }
        for other in self.siblings(tid) {
            self.tasks.remove(&other);
        }
        let mut task = self.tasks.remove(&tid).unwrap();
        task.tid = pid;
        {
            let mut core = task.core.borrow_mut();
            core.saved.clear();
            core.loaded = pid;
            core.vcpu.reset();
        }
        task.process.borrow_mut().clear_child_tid.clear();
        task.emulator = Emulator::default();
        task.state = State::Runnable;
        let loaded = load(&mut task, &exec.elf, &exec.path, &exec.argv, &exec.envp);
        self.tasks.insert(pid, task);
        if loaded.is_err() {
            self.terminate(pid, Termination::Killed(SIGSEGV));
        }
    }

    /// Reap a zombie child of `tid`'s process, or block until there is one.
    /// Returns false if it blocked.
    fn wait(&mut self, tid: u32, wait: Wait) -> bool {
        let pid = self.pid_of(tid);
        let children: Vec<(u32, State)> = self
            .tasks
            .iter()
            .filter(|(child, task)| {
                task.is_leader()
                    && task.process().ppid == pid
                    && wait.pid.is_none_or(|wanted| wanted == **child)
            })
            .map(|(&child, task)| (child, task.state))
            .collect();
        if children.is_empty() {
            self.finish(tid, ECHILD.wrapping_neg());
            return true;
        }
        let zombie = children.iter().find_map(|&(child, state)| match state {
            State::Zombie(termination) => Some((child, termination)),
            _ => None,
        });
        let Some((child, termination)) = zombie else {
            if !wait.nohang {
                self.tasks.get_mut(&tid).unwrap().state = State::Blocked;
                return false;
            }
            let cleared = match wait.report {
                Report::Siginfo(info) => self.write(tid, info, &[0; 128]),
                Report::Status(_) => true,
            };
            self.finish(tid, if cleared { 0 } else { EFAULT.wrapping_neg() });
            return true;
        };
        let (written, value) = match wait.report {
            Report::Status(0) => (true, child),
            Report::Status(status) => {
                let word = termination.wait_status().to_le_bytes();
                (self.write(tid, status, &word), child)
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
                (self.write(tid, info, &siginfo), 0)
            }
        };
        if !written {
            self.finish(tid, EFAULT.wrapping_neg());
            return true;
        }
        self.tasks.remove(&child);
        self.finish(tid, value);
        true
    }

    /// The processes `target` names for a signal from thread `sender`: a
    /// pid, a process group (0 for the sender's, -g for group g), or every
    /// process but init (-1).
    fn targets(&self, sender: u32, target: i32) -> Vec<u32> {
        match target {
            1.. if self.leader(target as u32).is_some() => vec![target as u32],
            1.. => Vec::new(),
            0 => self.group(self.tasks[&sender].process().pgid),
            -1 => self
                .tasks
                .iter()
                .filter(|&(&pid, task)| pid != INIT && task.is_leader())
                .map(|(&pid, _)| pid)
                .collect(),
            _ => self.group(target.unsigned_abs()),
        }
    }

    /// The processes in the process group `pgid`.
    fn group(&self, pgid: u32) -> Vec<u32> {
        let members = self
            .tasks
            .iter()
            .filter(|(_, task)| task.is_leader() && task.process().pgid == pgid);
        members.map(|(&pid, _)| pid).collect()
    }

    /// Raise `signal` in each of the processes `pids`. A process's signals
    /// are its threads', so one of them handles it. Zombies are past killing.
    fn raise(&mut self, pids: &[u32], signal: u32) {
        for &pid in pids {
            let Some(task) = self.tasks.get(&pid) else {
                continue;
            };
            if matches!(task.state, State::Zombie(_)) {
                continue;
            }
            let killed = task.process.borrow_mut().signals.raise(signal);
            if let Some(termination) = killed {
                self.terminate(pid, termination);
            }
        }
    }

    /// The `kill` system call of thread `sender`: send `signal` to `target`
    /// (see `targets`), and return to the sender.
    fn kill(&mut self, sender: u32, target: i32, signal: u32) {
        let targets = self.targets(sender, target);
        if targets.is_empty() {
            self.finish(sender, ESRCH.wrapping_neg());
            return;
        }
        self.raise(&targets, signal);
        if self
            .tasks
            .get(&sender)
            .is_some_and(|task| !matches!(task.state, State::Zombie(_)))
        {
            self.finish(sender, 0);
        }
    }

    /// A process other than init that faults is killed, as Linux would with
    /// SIGSEGV or SIGILL. A fault in init stops the container with the details.
    fn fault(&mut self, tid: u32, exit: Exit) -> Result<(), RunError> {
        if self.pid_of(tid) == INIT {
            return Err(RunError::Fault(exit));
        }
        self.faults.push((tid, exit));
        let signal = match exit.reason {
            ExitReason::IllegalInstruction(_) => SIGILL,
            ExitReason::Ebreak => SIGTRAP,
            _ => SIGSEGV,
        };
        self.terminate(tid, Termination::Killed(signal));
        Ok(())
    }

    /// End the process `tid` belongs to, all its threads with it: close its
    /// files, which closes pipe ends, and hand its children to init.
    fn terminate(&mut self, tid: u32, termination: Termination) {
        let pid = self.pid_of(tid);
        for other in self.siblings(pid) {
            self.tasks.remove(&other);
        }
        let task = self.tasks.get_mut(&pid).unwrap();
        task.state = State::Zombie(termination);
        {
            let vcpu = cpu(&task.core, pid);
            let mut process = task.process.borrow_mut();
            process.sync_shared(&*vcpu, None);
            process.release();
        }
        let ppid = task.process().ppid;
        if let Some(parent) = self.leader(ppid) {
            // SIGCHLD is ignored by default, so this only queues it for a handler.
            parent.process.borrow_mut().signals.raise(SIGCHLD);
        }
        for task in self.tasks.values() {
            if task.is_leader() && task.process().ppid == pid {
                task.process.borrow_mut().ppid = INIT;
            }
        }
    }

    fn write(&mut self, tid: u32, addr: u32, bytes: &[u8]) -> bool {
        self.tasks[&tid].vcpu().write_mem(addr, bytes).is_ok()
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
    let mut vcpu = task.vcpu();
    let image = elf::load(&mut *vcpu, elf)?;
    let stack_top = vcpu.mem_size();
    let sp = elf::setup_stack(&mut *vcpu, &image, stack_top, argv, envp)?;
    let mut process = task.process.borrow_mut();
    process.exec(image.brk, stack_top.saturating_sub(STACK_SIZE));
    process.set_command(argv, exe);
    vcpu.set_reg(SP, sp);
    vcpu.set_pc(image.entry);
    Ok(())
}

/// Print one system call the way `strace` would, more or less.
fn trace(tid: u32, [nr, args @ ..]: [u32; 5], action: &Action) {
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
    eprintln!("[{tid}] syscall {nr}({a0:#x}, {a1:#x}, {a2:#x}, {a3:#x}) -> {outcome}");
}

fn task_state<V>(tasks: &mut BTreeMap<u32, Task<V>>, pid: u32, state: State) {
    tasks.get_mut(&pid).unwrap().state = state;
}
