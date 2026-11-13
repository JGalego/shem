//! Synthetic `/proc` nodes from the process table and virtual clock.
//!
//! Nodes are generated on lookup; file contents are captured when opened.

use super::abi::*;
use super::{EPOCH_NS, NS_PER_SEC};

/// What `/proc` shows of one process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcEntry {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub sid: u32,
    /// `R`, `S` or `Z`, as in `ps`.
    pub state: u8,
    /// The arguments, each followed by a NUL, as in `/proc/PID/cmdline`.
    pub cmdline: Vec<u8>,
    /// The program's guest path.
    pub exe: Vec<u8>,
    pub cwd: Vec<u8>,
    /// Bytes of address space in use: image, heap, mappings and stack.
    pub vsize: u64,
    /// Instructions retired, which on the virtual clock is CPU time in ns.
    pub cpu_ns: u64,
    /// When the process started, in ns since boot.
    pub started_ns: u64,
}

/// The container as `/proc` sees it at one system call.
pub struct Snapshot<'a> {
    pub processes: &'a [ProcEntry],
    /// The calling process, which `/proc/self` names.
    pub me: u32,
    pub uptime_ns: u64,
    /// Time spent asleep, which is when the guest is idle.
    pub idle_ns: u64,
    pub mem_total: u64,
}

/// A path under `/proc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// A directory and its entries, with their `DT_` types.
    Dir(Vec<(Vec<u8>, u8)>),
    File(Vec<u8>),
    Link(Vec<u8>),
}

/// Ticks of `USER_HZ`, the unit of times in `/proc`.
const TICK_NS: u64 = NS_PER_SEC / 100;
const TOP: [&str; 7] = [
    "cpuinfo", "loadavg", "meminfo", "mounts", "stat", "uptime", "version",
];
const PER_PROCESS: [(&str, u8); 6] = [
    ("cmdline", DT_REG),
    ("comm", DT_REG),
    ("cwd", DT_LNK),
    ("exe", DT_LNK),
    ("stat", DT_REG),
    ("status", DT_REG),
];

/// Whether the absolute, normalized guest path `path` is under `/proc`.
pub fn is_proc(path: &[u8]) -> bool {
    path == b"/proc" || path.starts_with(b"/proc/")
}

/// Look up the path `path` under `/proc`. With `follow`, a final symlink
/// that points back into `/proc` is followed; one that leads into the root
/// filesystem, such as `exe`, is returned as the link.
pub fn lookup(path: &[u8], follow: bool, s: &Snapshot) -> Result<Node, Errno> {
    let parts: Vec<&[u8]> = path
        .split(|&b| b == b'/')
        .filter(|p| !p.is_empty())
        .skip(1)
        .collect();
    let me = s.me.to_string().into_bytes();
    let (first, rest) = match parts.split_first() {
        None => return Ok(Node::Dir(top_entries(s))),
        Some((first, rest)) => (*first, rest),
    };
    let pid_part: &[u8] = if first == b"self" {
        if rest.is_empty() && !follow {
            return Ok(Node::Link(me));
        }
        &me
    } else {
        first
    };
    if let Some(process) = std::str::from_utf8(pid_part)
        .ok()
        .and_then(|p| p.parse::<u32>().ok())
        .and_then(|pid| s.processes.iter().find(|p| p.pid == pid))
    {
        return match rest {
            [] => Ok(Node::Dir(
                PER_PROCESS
                    .iter()
                    .map(|(name, kind)| (name.as_bytes().to_vec(), *kind))
                    .collect(),
            )),
            [file] => process_file(file, process),
            _ => Err(Errno::ENOTDIR),
        };
    }
    match (std::str::from_utf8(first), rest) {
        (Ok(name), []) if TOP.contains(&name) => Ok(Node::File(top_file(name, s))),
        (Ok(name), [_, ..]) if TOP.contains(&name) => Err(Errno::ENOTDIR),
        _ => Err(Errno::ENOENT),
    }
}

fn top_entries(s: &Snapshot) -> Vec<(Vec<u8>, u8)> {
    let mut entries: Vec<(Vec<u8>, u8)> = s
        .processes
        .iter()
        .map(|p| (p.pid.to_string().into_bytes(), DT_DIR))
        .collect();
    entries.push((b"self".to_vec(), DT_LNK));
    entries.extend(TOP.iter().map(|name| (name.as_bytes().to_vec(), DT_REG)));
    entries
}

fn process_file(name: &[u8], p: &ProcEntry) -> Result<Node, Errno> {
    Ok(match name {
        b"cmdline" => Node::File(p.cmdline.clone()),
        b"comm" => Node::File([comm(p).as_slice(), b"\n"].concat()),
        b"cwd" => Node::Link(p.cwd.clone()),
        b"exe" => Node::Link(p.exe.clone()),
        b"stat" => Node::File(stat_line(p).into_bytes()),
        b"status" => Node::File(status(p).into_bytes()),
        _ => return Err(Errno::ENOENT),
    })
}

/// The command name: the program's file name, cut to 15 bytes, as Linux keeps it.
fn comm(p: &ProcEntry) -> Vec<u8> {
    let name = p.exe.rsplit(|&b| b == b'/').next().unwrap_or(&p.exe);
    name[..name.len().min(15)].to_vec()
}

fn pages(bytes: u64) -> u64 {
    bytes.div_ceil(4096)
}

fn stat_line(p: &ProcEntry) -> String {
    let comm = String::from_utf8_lossy(&comm(p)).into_owned();
    let utime = p.cpu_ns / TICK_NS;
    let start = p.started_ns / TICK_NS;
    // Fields 1 to 52 of proc(5): mostly zeros, as for a process that has
    // never faulted, been niced or had a terminal.
    format!(
        "{pid} ({comm}) {state} {ppid} {pgid} {sid} 0 -1 4194304 0 0 0 0 {utime} 0 0 0 20 0 1 0 {start} {vsize} {rss} 4294967295 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
        pid = p.pid,
        state = p.state as char,
        ppid = p.ppid,
        pgid = p.pgid,
        sid = p.sid,
        vsize = p.vsize,
        rss = pages(p.vsize),
    )
}

fn status(p: &ProcEntry) -> String {
    let state = match p.state {
        b'R' => "R (running)",
        b'Z' => "Z (zombie)",
        _ => "S (sleeping)",
    };
    format!(
        "Name:\t{name}\nState:\t{state}\nTgid:\t{pid}\nPid:\t{pid}\nPPid:\t{ppid}\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\nVmSize:\t{kb} kB\nVmRSS:\t{kb} kB\nThreads:\t1\n",
        name = String::from_utf8_lossy(&comm(p)),
        pid = p.pid,
        ppid = p.ppid,
        kb = p.vsize / 1024,
    )
}

fn top_file(name: &str, s: &Snapshot) -> Vec<u8> {
    let centis = |ns: u64| format!("{}.{:02}", ns / NS_PER_SEC, ns % NS_PER_SEC / TICK_NS);
    let running = s.processes.iter().filter(|p| p.state == b'R').count();
    let last_pid = s.processes.iter().map(|p| p.pid).max().unwrap_or(0);
    let used: u64 = s.processes.iter().map(|p| p.vsize).sum();
    let free = s.mem_total.saturating_sub(used);
    let busy = (s.processes.iter()).fold(0u64, |busy, p| busy.saturating_add(p.cpu_ns));
    match name {
        "uptime" => format!("{} {}\n", centis(s.uptime_ns), centis(s.idle_ns)),
        "loadavg" => format!(
            "0.00 0.00 0.00 {running}/{} {last_pid}\n",
            s.processes.len()
        ),
        "meminfo" => format!(
            "MemTotal:{:>16} kB\nMemFree:{:>17} kB\nMemAvailable:{:>12} kB\nBuffers:{:>17} kB\nCached:{:>18} kB\nSwapTotal:{:>15} kB\nSwapFree:{:>16} kB\n",
            s.mem_total / 1024,
            free / 1024,
            free / 1024,
            0,
            0,
            0,
            0
        ),
        "cpuinfo" => "processor\t: 0\nhart\t\t: 0\nisa\t\t: rv32ima\nmmu\t\t: none\nuarch\t\t: shem\n\n".into(),
        "version" => "Linux version 6.6.0-shem (shem) #1\n".into(),
        "mounts" => "shem / shem rw 0 0\nproc /proc proc rw,nosuid,nodev,noexec 0 0\n".into(),
        "stat" => {
            let (user, idle) = (busy / TICK_NS, s.idle_ns / TICK_NS);
            format!(
                "cpu  {user} 0 0 {idle} 0 0 0 0 0 0\ncpu0 {user} 0 0 {idle} 0 0 0 0 0 0\nbtime {}\nprocesses {last_pid}\nprocs_running {running}\nprocs_blocked 0\n",
                EPOCH_NS / NS_PER_SEC
            )
        }
        _ => unreachable!("not a top-level /proc file: {name}"),
    }
    .into_bytes()
}

/// The status `stat` reports for a node at `path`.
pub fn stat(path: &[u8], node: &Node) -> Stat {
    // Inode numbers far above the root filesystem's, from the path
    let ino = path.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| {
        (hash ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    }) | 1 << 62;
    let (mode, nlink, size) = match node {
        Node::Dir(_) => (S_IFDIR | 0o555, 2, 0),
        Node::File(_) => (S_IFREG | 0o444, 1, 0),
        Node::Link(target) => (S_IFLNK | 0o777, 1, target.len() as u64),
    };
    let boot = ((EPOCH_NS / NS_PER_SEC) as i64, 0);
    Stat {
        mode,
        nlink,
        ino,
        size,
        blocks: 0,
        blksize: 1024,
        atime: boot,
        mtime: boot,
        ctime: boot,
        dev: PROC_DEV,
        rdev: 0,
    }
}

/// `struct sysinfo` for an RV32 guest, 64 bytes.
pub fn sysinfo(s: &Snapshot) -> [u8; 64] {
    let used: u64 = s.processes.iter().map(|p| p.vsize).sum();
    let mut out = [0u8; 64];
    let mut put = |at: usize, value: u32| out[at..at + 4].copy_from_slice(&value.to_le_bytes());
    put(0, (s.uptime_ns / NS_PER_SEC) as u32);
    put(16, s.mem_total as u32); // totalram
    put(20, s.mem_total.saturating_sub(used) as u32); // freeram
    out[40..42].copy_from_slice(&(s.processes.len() as u16).to_le_bytes());
    out[52..56].copy_from_slice(&1u32.to_le_bytes()); // mem_unit
    out
}
