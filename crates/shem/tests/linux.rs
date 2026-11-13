//! System calls, exercised directly: set the argument registers, call
//! `Linux::handle`, and check the result and guest memory.

mod common;

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use common::SharedBuf;
use shem::linux::{Action, Exec, Linux, ProcEntry, Process, Report, Stdio, Termination, Wait};
use shem::reference::ReferenceVcpu;
use shem::vcpu::{A0, A7, Vcpu};
use tempfile::TempDir;

// asm-generic numbers and flags, written out independently of the implementation
const GETCWD: u32 = 17;
const DUP: u32 = 23;
const FCNTL64: u32 = 25;
const MKDIRAT: u32 = 34;
const UNLINKAT: u32 = 35;
const SYMLINKAT: u32 = 36;
const CHDIR: u32 = 49;
const OPENAT: u32 = 56;
const CLOSE: u32 = 57;
const PIPE2: u32 = 59;
const GETDENTS64: u32 = 61;
const LLSEEK: u32 = 62;
const READ: u32 = 63;
const WRITE: u32 = 64;
const READV: u32 = 65;
const READLINKAT: u32 = 78;
const EXIT: u32 = 93;
const EXIT_GROUP: u32 = 94;
const KILL: u32 = 129;
const RT_SIGACTION: u32 = 134;
const UNAME: u32 = 160;
const STATFS64: u32 = 43;
const SYSINFO: u32 = 179;
const CLONE: u32 = 220;
const MMAP: u32 = 222;
const MUNMAP: u32 = 215;
const MSYNC: u32 = 227;
const IOCTL: u32 = 29;
const PPOLL_TIME64: u32 = 414;
const EXECVE: u32 = 221;
const WAIT4: u32 = 260;
const RENAMEAT2: u32 = 276;
const GETRANDOM: u32 = 278;
const STATX: u32 = 291;
const CLOCK_GETTIME64: u32 = 403;
const CLOCK_NANOSLEEP_TIME64: u32 = 407;

const AT_FDCWD: u32 = -100i32 as u32;
const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
const AT_REMOVEDIR: u32 = 0x200;
const AT_EMPTY_PATH: u32 = 0x1000;
const O_RDONLY: u32 = 0;
const O_WRONLY: u32 = 1;
const O_RDWR: u32 = 2;
const O_CREAT: u32 = 0o100;
const O_EXCL: u32 = 0o200;
const O_NONBLOCK: u32 = 0o4000;
const O_TRUNC: u32 = 0o1000;
const O_DIRECTORY: u32 = 0o200000;
const O_CLOEXEC: u32 = 0o2000000;
const F_GETFD: u32 = 1;
const RENAME_NOREPLACE: u32 = 1;
const CLOCK_REALTIME: u32 = 0;
const CLOCK_MONOTONIC: u32 = 1;
const SIGCHLD: u32 = 17;
const SIGPIPE: u32 = 13;
const CLONE_VM: u32 = 0x100;
const CLONE_THREAD: u32 = 0x10000;
const WNOHANG: u32 = 1;
const PROT_READ: u32 = 1;
const PROT_WRITE: u32 = 2;
const MAP_SHARED: u32 = 1;
const MAP_PRIVATE: u32 = 2;

const ENOENT: i32 = -2;
const EBADF: i32 = -9;
const EEXIST: i32 = -17;
const ENOTDIR: i32 = -20;
const EISDIR: i32 = -21;
const EINVAL: i32 = -22;
const EACCES: i32 = -13;
const ENOEXEC: i32 = -8;
const EAGAIN: i32 = -11;
const EPIPE: i32 = -32;
const EBUSY: i32 = -16;
const ENODEV: i32 = -19;
const ENOTTY: i32 = -25;

const MEM: u32 = 1 << 20;

struct Guest {
    linux: Linux,
    process: Process,
    vcpu: ReferenceVcpu,
    stdout: SharedBuf,
    root: TempDir,
    scratch: u32,
    retired: u64,
    /// The process table the container would hand `/proc`.
    procs: Vec<ProcEntry>,
}

impl Guest {
    fn new() -> Self {
        let mut guest = Self::without_root();
        guest.process.set_root(guest.root.path()).unwrap();
        guest
    }

    fn without_root() -> Self {
        let stdout = SharedBuf::default();
        let stdio = Stdio {
            stdin: Box::new(Cursor::new(b"typed input".to_vec())),
            stdout: Box::new(stdout.clone()),
            stderr: Box::new(std::io::sink()),
        };
        Self {
            linux: Linux::new(),
            process: Process::new(1, stdio, 0x1000, MEM),
            vcpu: ReferenceVcpu::new(MEM),
            stdout,
            root: tempfile::tempdir().unwrap(),
            scratch: 0x8_0000,
            retired: 0,
            procs: Vec::new(),
        }
    }

    fn host(&self, path: &str) -> PathBuf {
        self.root.path().join(path)
    }

    /// Copy `bytes` into guest memory and return their address.
    fn put(&mut self, bytes: &[u8]) -> u32 {
        let addr = self.scratch;
        self.vcpu.write_mem(addr, bytes).unwrap();
        self.scratch += (bytes.len() as u32).max(1).next_multiple_of(8);
        addr
    }

    fn cstr(&mut self, s: &str) -> u32 {
        self.put(&[s.as_bytes(), &[0]].concat())
    }

    fn buf(&mut self, len: u32) -> u32 {
        self.put(&vec![0; len as usize])
    }

    fn get(&self, addr: u32, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        self.vcpu.read_mem(addr, &mut bytes).unwrap();
        bytes
    }

    fn action(&mut self, nr: u32, args: &[u32]) -> Action {
        self.vcpu.set_reg(A7, nr);
        for (i, &arg) in args.iter().enumerate() {
            self.vcpu.set_reg(A0 + i, arg);
        }
        self.linux.processes = self.procs.clone();
        self.linux
            .handle(&mut self.process, &mut self.vcpu, self.retired)
    }

    fn call(&mut self, nr: u32, args: &[u32]) -> i32 {
        match self.action(nr, args) {
            Action::Return(value) => value as i32,
            action => panic!("expected a return value, got {action:?}"),
        }
    }

    fn pipe(&mut self, flags: u32) -> (u32, u32) {
        let fds = self.buf(8);
        assert_eq!(self.call(PIPE2, &[fds, flags]), 0);
        let fds = self.get(fds, 8);
        (word(&fds, 0), word(&fds, 4))
    }

    /// Write a NULL-terminated array of strings into guest memory.
    fn strings(&mut self, list: &[&str]) -> u32 {
        let pointers: Vec<u32> = list.iter().map(|s| self.cstr(s)).chain([0]).collect();
        self.put(
            &pointers
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    }

    fn open(&mut self, path: &str, flags: u32) -> i32 {
        let path = self.cstr(path);
        self.call(OPENAT, &[AT_FDCWD, path, flags, 0o644])
    }

    fn read(&mut self, fd: i32, len: u32) -> Vec<u8> {
        let buf = self.buf(len);
        let n = self.call(READ, &[fd as u32, buf, len]);
        assert!(n >= 0, "read failed: {n}");
        self.get(buf, n as usize)
    }

    fn write(&mut self, fd: i32, data: &[u8]) -> i32 {
        let buf = self.put(data);
        self.call(WRITE, &[fd as u32, buf, data.len() as u32])
    }

    fn statx(&mut self, dirfd: u32, path: &str, flags: u32) -> (i32, Vec<u8>) {
        let path = self.cstr(path);
        let buf = self.buf(256);
        let result = self.call(STATX, &[dirfd, path, flags, 0x7ff, buf]);
        (result, self.get(buf, 256))
    }
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn dword(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[test]
fn files_round_trip_through_the_host() {
    let mut g = Guest::new();
    let fd = g.open("/notes.txt", O_WRONLY | O_CREAT | O_TRUNC);
    assert_eq!(fd, 3);
    assert_eq!(g.write(fd, b"hello"), 5);
    assert_eq!(g.call(CLOSE, &[fd as u32]), 0);
    assert_eq!(fs::read(g.host("notes.txt")).unwrap(), b"hello");

    let fd = g.open("/notes.txt", O_RDONLY);
    let result = g.buf(8);
    assert_eq!(g.call(LLSEEK, &[fd as u32, 0, 1, result, 0]), 0);
    assert_eq!(dword(&g.get(result, 8), 0), 1);
    assert_eq!(g.read(fd, 16), b"ello");
}

#[test]
fn open_reports_linux_errors() {
    let mut g = Guest::new();
    fs::write(g.host("file"), b"").unwrap();
    fs::create_dir(g.host("dir")).unwrap();
    assert_eq!(g.open("/missing", O_RDONLY), ENOENT);
    assert_eq!(g.open("/file", O_WRONLY | O_CREAT | O_EXCL), EEXIST);
    assert_eq!(g.open("/dir", O_WRONLY), EISDIR);
    assert_eq!(g.open("/file", O_RDONLY | O_DIRECTORY), ENOTDIR);
    let fd = g.open("/file", O_RDONLY);
    assert_eq!(g.write(fd, b"x"), EBADF);
    assert_eq!(g.write(99, b"x"), EBADF);
}

#[test]
fn getdents64_lists_entries_sorted_with_types() {
    let mut g = Guest::new();
    fs::write(g.host("b"), b"").unwrap();
    fs::write(g.host("a"), b"").unwrap();
    fs::create_dir(g.host("c")).unwrap();
    let fd = g.open("/", O_RDONLY | O_DIRECTORY) as u32;
    let entries = list(&mut g, fd);
    // /proc is listed even though the rootfs has no such directory
    let expected = [
        (".", 4),
        ("..", 4),
        ("a", 8),
        ("b", 8),
        ("c", 4),
        ("proc", 4),
    ];
    let expected: Vec<(String, u8)> = expected.iter().map(|&(n, t)| (n.into(), t)).collect();
    assert_eq!(entries, expected);
    let buf = g.buf(8);
    assert_eq!(g.call(GETDENTS64, &[fd, buf, 4096]), 0);
    assert_eq!(g.call(GETDENTS64, &[fd, buf, 8]), 0);
}

/// The names and types a directory open on `fd` lists, from one getdents64.
fn list(g: &mut Guest, fd: u32) -> Vec<(String, u8)> {
    let buf = g.buf(4096);
    let n = g.call(GETDENTS64, &[fd, buf, 4096]);
    let bytes = g.get(buf, n as usize);
    let mut entries = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let len = u16::from_le_bytes([bytes[at + 16], bytes[at + 17]]) as usize;
        let name = &bytes[at + 19..at + len];
        let name = &name[..name.iter().position(|&b| b == 0).unwrap()];
        entries.push((String::from_utf8(name.to_vec()).unwrap(), bytes[at + 18]));
        at += len;
    }
    entries
}

#[test]
fn proc_shows_the_processes_and_the_virtual_clock() {
    let mut g = Guest::new();
    g.process.set_command(&["/bin/sh", "-c", "ps"], b"/bin/sh");
    let init = g.process.proc_entry(b'R');
    let child = ProcEntry {
        pid: 2,
        ppid: 1,
        state: b'S',
        cmdline: b"sleep\x0010\x00".to_vec(),
        exe: b"/bin/sleep".to_vec(),
        cpu_ns: 2_000_000_000,
        started_ns: 1_000_000_000,
        ..init.clone()
    };
    g.procs = vec![init, child];
    g.retired = 3_250_000_000;

    let fd = g.open("/proc", O_RDONLY | O_DIRECTORY) as u32;
    let names: Vec<String> = list(&mut g, fd).into_iter().map(|(name, _)| name).collect();
    for name in [
        "1", "2", "self", "uptime", "meminfo", "mounts", "cpuinfo", "stat",
    ] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
    let read = |g: &mut Guest, path: &str| {
        let fd = g.open(path, O_RDONLY);
        assert!(fd >= 0, "{path}: {fd}");
        String::from_utf8(g.read(fd, 4096)).unwrap()
    };
    assert_eq!(read(&mut g, "/proc/uptime"), "3.25 0.00\n");
    assert_eq!(read(&mut g, "/proc/self/cmdline"), "/bin/sh\0-c\0ps\0");
    assert!(
        read(&mut g, "/proc/2/stat")
            .starts_with("2 (sleep) S 1 1 1 0 -1 4194304 0 0 0 0 200 0 0 0 20 0 1 0 100 ")
    );
    assert!(read(&mut g, "/proc/2/status").contains("Name:\tsleep\nState:\tS (sleeping)\n"));
    assert!(
        read(&mut g, "/proc/meminfo").starts_with(&format!("MemTotal:{:>16} kB\n", MEM / 1024))
    );
    assert!(read(&mut g, "/proc/mounts").starts_with("shem / shem rw"));

    let readlink = |g: &mut Guest, path: &str| {
        let (path, buf) = (g.cstr(path), g.buf(64));
        let n = g.call(READLINKAT, &[AT_FDCWD, path, buf, 64]);
        String::from_utf8(g.get(buf, n.max(0) as usize)).unwrap()
    };
    assert_eq!(readlink(&mut g, "/proc/self"), "1");
    assert_eq!(readlink(&mut g, "/proc/2/exe"), "/bin/sleep");

    let (result, st) = g.statx(AT_FDCWD, "/proc/2", 0);
    assert_eq!(
        (result, word(&st, 28) & 0o170000),
        (0, 0o040000),
        "a directory"
    );
    let (_, st) = g.statx(AT_FDCWD, "/proc/uptime", 0);
    assert_eq!(word(&st, 28) & 0o177777, 0o100444, "a read-only file");
    assert_eq!(g.open("/proc/uptime", O_WRONLY), EACCES);
    assert_eq!(g.open("/proc/3/stat", O_RDONLY), ENOENT);

    let info = g.buf(64);
    assert_eq!(g.call(SYSINFO, &[info]), 0);
    let info = g.get(info, 64);
    assert_eq!(
        (word(&info, 0), word(&info, 16)),
        (3, MEM),
        "uptime and RAM"
    );
    assert_eq!(u16::from_le_bytes([info[40], info[41]]), 2, "processes");

    let (root, proc_dir, buf) = (g.cstr("/"), g.cstr("/proc"), g.buf(88));
    assert_eq!(g.call(STATFS64, &[root, 88, buf]), 0);
    assert_eq!(
        (word(&g.get(buf, 88), 0), word(&g.get(buf, 88), 4)),
        (0x5348_454d, 4096)
    );
    assert_eq!(g.call(STATFS64, &[proc_dir, 88, buf]), 0);
    assert_eq!(word(&g.get(buf, 88), 0), 0x9fa0);

    let path = g.cstr("/proc/self");
    assert_eq!(g.call(CHDIR, &[path]), 0);
    assert!(read(&mut g, "stat").starts_with("1 (sh) R 0 1 1 "));
}

#[test]
fn getdents64_needs_room_for_one_entry() {
    let mut g = Guest::new();
    let fd = g.open("/", O_RDONLY | O_DIRECTORY) as u32;
    let buf = g.buf(8);
    assert_eq!(g.call(GETDENTS64, &[fd, buf, 8]), EINVAL);
}

#[test]
fn statx_reports_files_links_and_stdio() {
    let mut g = Guest::new();
    fs::write(g.host("file"), b"hello").unwrap();
    std::os::unix::fs::symlink("file", g.host("link")).unwrap();

    let (result, st) = g.statx(AT_FDCWD, "/link", 0);
    assert_eq!(result, 0);
    assert_eq!(dword(&st, 40), 5, "size");
    assert_eq!(word(&st, 28) & 0xffff & 0o170000, 0o100000, "regular file");
    assert_eq!((word(&st, 20), word(&st, 24)), (0, 0), "owned by root");

    let (_, st) = g.statx(AT_FDCWD, "/link", AT_SYMLINK_NOFOLLOW);
    assert_eq!(word(&st, 28) & 0o170000, 0o120000, "symlink");

    let (result, st) = g.statx(1, "", AT_EMPTY_PATH);
    assert_eq!(result, 0);
    assert_eq!(word(&st, 28) & 0o170000, 0o020000, "character device");
}

#[test]
fn file_metadata_follows_the_virtual_clock_not_the_host() {
    const EPOCH_S: u64 = 1_767_225_600; // 2026-01-01, the virtual clock's start
    let mut g = Guest::new();
    g.retired = 1_000;
    let fd = g.open("/new", O_WRONLY | O_CREAT);
    g.retired = 5_000;
    assert_eq!(g.write(fd, b"hello"), 5);

    let (_, st) = g.statx(AT_FDCWD, "/new", 0);
    let time = |st: &[u8], at: usize| (dword(st, at), word(st, at + 8));
    assert_eq!(time(&st, 64), (EPOCH_S, 1_000), "accessed: when created");
    assert_eq!(time(&st, 112), (EPOCH_S, 5_000), "modified: when written");
    assert_eq!(time(&st, 96), (EPOCH_S, 5_000), "changed: as modified");
    let (_, root) = g.statx(AT_FDCWD, "/", 0);
    assert_eq!(
        time(&root, 112),
        (EPOCH_S, 1_000),
        "the directory, when the file appeared"
    );

    // Inode numbers count up as files are first seen; the device and the
    // block count are fixed or computed
    assert_eq!((dword(&st, 32), dword(&root, 32)), (1, 2));
    assert_eq!(g.statx(AT_FDCWD, "/new", 0).1, st);
    assert_eq!((word(&st, 136), word(&st, 140)), (8, 1), "device 8:1");
    assert_eq!((word(&st, 4), dword(&st, 48)), (4096, 8), "one 4 KiB block");

    g.retired = 9_000;
    let path = g.cstr("/new");
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, path, 0]), 0);
    let (_, root) = g.statx(AT_FDCWD, "/", 0);
    assert_eq!(
        time(&root, 112),
        (EPOCH_S, 9_000),
        "the directory, when the file went"
    );
}

#[test]
fn working_directory_and_relative_paths() {
    let mut g = Guest::new();
    let buf = g.buf(64);
    assert_eq!(g.call(GETCWD, &[buf, 64]), 2);
    assert_eq!(g.get(buf, 2), b"/\0");

    let work = g.cstr("work");
    assert_eq!(g.call(MKDIRAT, &[AT_FDCWD, work, 0o755]), 0);
    assert_eq!(g.call(CHDIR, &[work]), 0);
    assert_eq!(g.call(GETCWD, &[buf, 64]), 6);
    assert_eq!(g.get(buf, 6), b"/work\0");

    let fd = g.open("f", O_WRONLY | O_CREAT);
    assert!(fd >= 0);
    assert!(g.host("work/f").is_file());
    let nope = g.cstr("/nope");
    assert_eq!(g.call(CHDIR, &[nope]), ENOENT);
}

#[test]
fn links_renames_and_removal() {
    let mut g = Guest::new();
    fs::write(g.host("a"), b"").unwrap();
    fs::write(g.host("b"), b"").unwrap();
    fs::create_dir(g.host("dir")).unwrap();

    let (target, link) = (g.cstr("/a"), g.cstr("link"));
    assert_eq!(g.call(SYMLINKAT, &[target, AT_FDCWD, link]), 0);
    let buf = g.buf(64);
    assert_eq!(g.call(READLINKAT, &[AT_FDCWD, link, buf, 64]), 2);
    assert_eq!(g.get(buf, 2), b"/a");

    let (a, b, c) = (g.cstr("a"), g.cstr("b"), g.cstr("c"));
    let noreplace = [AT_FDCWD, a, AT_FDCWD, b, RENAME_NOREPLACE];
    assert_eq!(g.call(RENAMEAT2, &noreplace), EEXIST);
    assert_eq!(g.call(RENAMEAT2, &[AT_FDCWD, a, AT_FDCWD, c, 0]), 0);
    assert!(g.host("c").is_file() && !g.host("a").exists());

    let (dir, root) = (g.cstr("dir"), g.cstr("/"));
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, dir, 0]), EISDIR);
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, dir, AT_REMOVEDIR]), 0);
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, c, 0]), 0);
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, root, AT_REMOVEDIR]), EBUSY);
    assert!(!g.host("dir").exists() && !g.host("c").exists());
}

#[test]
fn guest_paths_never_reach_host_files_outside_the_root() {
    let mut g = Guest::new();
    let outside = tempfile::NamedTempFile::new().unwrap();
    let outside = fs::canonicalize(outside.path()).unwrap();
    let outside = outside.to_str().unwrap();
    std::os::unix::fs::symlink(outside, g.host("escape")).unwrap();

    assert_eq!(
        g.open(&format!("/../../../../..{outside}"), O_RDONLY),
        ENOENT
    );
    assert_eq!(g.open("/escape", O_RDONLY), ENOENT);
    // Writing through the link creates a file inside the root, once the
    // directories the link names exist there; the host file is untouched.
    let inside = g.host(&outside[1..]);
    fs::create_dir_all(inside.parent().unwrap()).unwrap();
    let fd = g.open("/escape", O_WRONLY | O_CREAT);
    assert_eq!(g.write(fd, b"guest data"), 10);
    assert_eq!(fs::read(&inside).unwrap(), b"guest data");
    assert_eq!(fs::read(outside).unwrap(), b"");
}

#[test]
fn a_directory_swapped_for_a_link_after_the_guest_got_there_stays_inside_the_root() {
    let mut g = Guest::new();
    let outside = tempfile::tempdir().unwrap();
    let outside = fs::canonicalize(outside.path()).unwrap();
    fs::write(outside.join("secret"), b"host data").unwrap();
    let target = g.cstr(outside.to_str().unwrap());

    // The working directory: entered, removed, and replaced by a link that leads out
    let a = g.cstr("/a");
    assert_eq!(g.call(MKDIRAT, &[AT_FDCWD, a, 0o755]), 0);
    assert_eq!(g.call(CHDIR, &[a]), 0);
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, a, AT_REMOVEDIR]), 0);
    assert_eq!(g.call(SYMLINKAT, &[target, AT_FDCWD, a]), 0);
    assert_eq!(g.open("secret", O_RDONLY), ENOENT);
    assert_eq!(g.open("planted", O_WRONLY | O_CREAT), ENOENT);
    let secret = g.cstr("secret");
    assert_eq!(g.call(UNLINKAT, &[AT_FDCWD, secret, 0]), ENOENT);

    // A directory held open, renamed away, and replaced the same way
    let d = g.cstr("/d");
    assert_eq!(g.call(MKDIRAT, &[AT_FDCWD, d, 0o755]), 0);
    let dirfd = g.open("/d", O_RDONLY | O_DIRECTORY) as u32;
    let moved = g.cstr("/d.old");
    assert_eq!(g.call(RENAMEAT2, &[AT_FDCWD, d, AT_FDCWD, moved, 0]), 0);
    assert_eq!(g.call(SYMLINKAT, &[target, AT_FDCWD, d]), 0);
    assert_eq!(g.call(OPENAT, &[dirfd, secret, O_RDONLY, 0]), ENOENT);

    assert_eq!(fs::read(outside.join("secret")).unwrap(), b"host data");
    assert!(!outside.join("planted").exists());
}

#[test]
fn files_map_as_private_copies() {
    let mut g = Guest::new();
    let data: Vec<u8> = (0..6000u32).map(|i| (i % 251) as u8).collect();
    fs::write(g.host("data"), &data).unwrap();
    let fd = g.open("/data", O_RDONLY) as u32;

    // From the second page: the rest of the file, then zeros
    let private = PROT_READ | PROT_WRITE;
    let at = g.call(MMAP, &[0, 8192, private, MAP_PRIVATE, fd, 1]) as u32;
    assert_eq!(g.get(at, 6000 - 4096), &data[4096..]);
    assert_eq!(g.get(at + 6000 - 4096, 16), [0; 16]);
    g.vcpu.write_mem(at, b"changed").unwrap();
    assert_eq!(
        fs::read(g.host("data")).unwrap(),
        data,
        "writes stay private"
    );

    let shared = g.call(MMAP, &[0, 4096, PROT_READ, MAP_SHARED, fd, 0]) as u32;
    assert_eq!(g.get(shared, 4096), &data[..4096]);
    let writable = [0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0];
    assert_eq!(
        g.call(MMAP, &writable),
        EACCES,
        "a shared writable mapping needs O_RDWR"
    );

    let write_only = g.open("/data", O_WRONLY) as u32;
    assert_eq!(
        g.call(MMAP, &[0, 4096, PROT_READ, MAP_PRIVATE, write_only, 0]),
        EACCES
    );
    assert_eq!(
        g.call(MMAP, &[0, 4096, PROT_READ, MAP_PRIVATE, 0, 0]),
        ENODEV,
        "stdin"
    );
}

#[test]
fn shared_mappings_write_back_to_their_files() {
    let mut g = Guest::new();
    fs::write(g.host("data"), vec![b'.'; 6000]).unwrap();
    let fd = g.open("/data", O_RDWR) as u32;
    let shared = [0, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0];
    let at = g.call(MMAP, &shared) as u32;

    g.vcpu.write_mem(at + 10, b"synced").unwrap();
    assert_eq!(
        &fs::read(g.host("data")).unwrap()[10..16],
        b"......",
        "not before msync"
    );
    assert_eq!(g.call(MSYNC, &[at, 4096, 4]), 0);
    assert_eq!(&fs::read(g.host("data")).unwrap()[10..16], b"synced");

    // munmap writes back too; and only the file's length, which never grows
    g.vcpu.write_mem(at + 5998, b"edge!").unwrap();
    assert_eq!(g.call(MUNMAP, &[at, 8192]), 0);
    let data = fs::read(g.host("data")).unwrap();
    assert_eq!((data.len(), &data[5998..]), (6000, &b"ed"[..]));

    // A process that exits with a mapping writes it back when it is released
    let at = g.call(MMAP, &shared) as u32;
    g.vcpu.write_mem(at, b"exit").unwrap();
    g.process.sync_shared(&g.vcpu, None);
    assert_eq!(&fs::read(g.host("data")).unwrap()[..4], b"exit");
}

/// A pseudo-terminal: the host's side, and the side the guest gets.
#[cfg(target_os = "linux")]
fn pty() -> (std::fs::File, std::fs::File) {
    use std::os::fd::FromRawFd;
    // SAFETY: the usual posix_openpt sequence, on descriptors this test owns
    unsafe {
        let master = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
        assert!(master >= 0);
        assert_eq!(libc::grantpt(master), 0);
        assert_eq!(libc::unlockpt(master), 0);
        let mut name = [0 as libc::c_char; 128];
        assert_eq!(libc::ptsname_r(master, name.as_mut_ptr(), name.len()), 0);
        let slave = libc::open(name.as_ptr(), libc::O_RDWR | libc::O_NOCTTY);
        assert!(slave >= 0);
        (
            std::fs::File::from_raw_fd(master),
            std::fs::File::from_raw_fd(slave),
        )
    }
}

#[test]
#[cfg(target_os = "linux")]
fn terminals_pass_through_to_the_host_terminal() {
    use std::io::Write;
    use std::os::fd::AsRawFd;

    let mut g = Guest::new();
    let buf = g.buf(64);
    assert_eq!(
        g.call(IOCTL, &[0, 0x5401, buf]),
        ENOTTY,
        "no terminal: not a tty"
    );

    let (mut host, guest_side) = pty();
    g.linux.terminal = Some(guest_side.as_raw_fd());
    assert_eq!(g.call(IOCTL, &[0, 0x5401, buf]), 0, "TCGETS");
    let termios = g.get(buf, 36);
    // SAFETY: tcgetattr into a zeroed termios
    let lflag = unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        libc::tcgetattr(guest_side.as_raw_fd(), &mut t);
        t.c_lflag
    };
    assert_eq!(word(&termios, 12), lflag, "the host terminal's settings");

    // Settings and window size go to the host terminal
    let mut raw = termios.clone();
    raw[12..16].copy_from_slice(&(lflag & !(libc::ICANON | libc::ECHO)).to_le_bytes());
    let raw = g.put(&raw);
    assert_eq!(g.call(IOCTL, &[1, 0x5402, raw]), 0, "TCSETS");
    let size = g.put(&[30, 0, 100, 0, 0, 0, 0, 0]);
    assert_eq!(g.call(IOCTL, &[1, 0x5414, size]), 0, "TIOCSWINSZ");
    assert_eq!(g.call(IOCTL, &[2, 0x5413, buf]), 0, "TIOCGWINSZ");
    assert_eq!(g.get(buf, 4), [30, 0, 100, 0]);
    // SAFETY: as above
    let now = unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        libc::tcgetattr(guest_side.as_raw_fd(), &mut t);
        t.c_lflag
    };
    assert_eq!(now & (libc::ICANON | libc::ECHO), 0, "raw mode on the host");

    // The foreground process group is the container's
    let group = g.put(&7u32.to_le_bytes());
    assert_eq!(g.call(IOCTL, &[0, 0x5410, group]), 0, "TIOCSPGRP");
    assert_eq!(g.call(IOCTL, &[0, 0x540f, buf]), 0, "TIOCGPGRP");
    assert_eq!(word(&g.get(buf, 4), 0), 7);

    // ppoll waits on the host terminal: nothing typed, then a key
    let fds = g.put(&[0, 0, 0, 0, 1, 0, 0, 0]); // fd 0, POLLIN
    let zero = g.put(&[0; 16]);
    assert_eq!(g.call(PPOLL_TIME64, &[fds, 1, zero, 0, 8]), 0);
    host.write_all(b"x").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert_eq!(g.call(PPOLL_TIME64, &[fds, 1, zero, 0, 8]), 1);
    assert_eq!(&g.get(fds, 8)[6..8], &[1, 0], "POLLIN");
}

#[test]
fn ppoll_reports_pipes_and_sleeps_through_timeouts() {
    let mut g = Guest::new();
    let (read_end, write_end) = g.pipe(0);
    let fds = g.put(
        &[
            &read_end.to_le_bytes()[..],
            &[1, 0, 0, 0],
            &write_end.to_le_bytes(),
            &[4, 0, 0, 0],
        ]
        .concat(),
    );
    let second = g.put(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // The empty pipe can be written but not read
    assert_eq!(g.call(PPOLL_TIME64, &[fds, 2, second, 0, 8]), 1);
    assert_eq!(&g.get(fds, 16)[6..8], &[0, 0]);
    assert_eq!(&g.get(fds, 16)[14..16], &[4, 0], "POLLOUT");
    assert_eq!(g.write(write_end as i32, b"hi"), 2);
    assert_eq!(g.call(PPOLL_TIME64, &[fds, 2, second, 0, 8]), 2);
    // With nothing ready and no timeout, the call waits to be retried
    let read_only = g.put(&[&read_end.to_le_bytes()[..], &[1, 0, 0, 0]].concat());
    g.read(read_end as i32, 2);
    assert_eq!(
        g.action(PPOLL_TIME64, &[read_only, 1, 0, 0, 8]),
        Action::Block
    );
}

#[test]
fn dup_shares_the_file_offset() {
    let mut g = Guest::new();
    fs::write(g.host("abc"), b"abcdef").unwrap();
    let fd = g.open("/abc", O_RDONLY | O_CLOEXEC);
    let copy = g.call(DUP, &[fd as u32]);
    assert_eq!(g.read(fd, 2), b"ab");
    assert_eq!(g.read(copy, 2), b"cd");
    assert_eq!(g.call(FCNTL64, &[fd as u32, F_GETFD, 0]), 1);
    assert_eq!(g.call(FCNTL64, &[copy as u32, F_GETFD, 0]), 0);
}

#[test]
fn readv_reads_stdin_across_buffers() {
    let mut g = Guest::new();
    let (first, second) = (g.buf(5), g.buf(32));
    let iov = g.put(&[first, 5, second, 32].map(u32::to_le_bytes).concat());
    assert_eq!(g.call(READV, &[0, iov, 2]), 11);
    assert_eq!(g.get(first, 5), b"typed");
    assert_eq!(g.get(second, 6), b" input");
}

#[test]
fn stdout_goes_to_the_host_stream() {
    let mut g = Guest::new();
    assert_eq!(g.write(1, b"out"), 3);
    assert_eq!(g.stdout.contents(), b"out");
}

#[test]
fn the_clock_is_virtual() {
    let mut g = Guest::new();
    let ts = g.buf(16);
    g.retired = 5_000;
    assert_eq!(g.call(CLOCK_GETTIME64, &[CLOCK_MONOTONIC, ts]), 0);
    assert_eq!(
        (dword(&g.get(ts, 16), 0), dword(&g.get(ts, 16), 8)),
        (0, 5_000)
    );

    let sleep = g.put(&[1i64.to_le_bytes(), 500_000_000i64.to_le_bytes()].concat());
    assert_eq!(
        g.call(CLOCK_NANOSLEEP_TIME64, &[CLOCK_MONOTONIC, 0, sleep, 0]),
        0
    );
    g.call(CLOCK_GETTIME64, &[CLOCK_MONOTONIC, ts]);
    assert_eq!(
        (dword(&g.get(ts, 16), 0), dword(&g.get(ts, 16), 8)),
        (1, 500_005_000)
    );

    g.call(CLOCK_GETTIME64, &[CLOCK_REALTIME, ts]);
    assert_eq!(
        dword(&g.get(ts, 16), 0),
        1_767_225_600 + 1,
        "2026-01-01 plus elapsed"
    );
}

#[test]
fn randomness_is_seeded_and_repeatable() {
    let bytes = |g: &mut Guest| {
        let buf = g.buf(32);
        assert_eq!(g.call(GETRANDOM, &[buf, 32, 0]), 32);
        g.get(buf, 32)
    };
    let (mut first, mut second) = (Guest::new(), Guest::new());
    let sample = bytes(&mut first);
    assert_eq!(sample, bytes(&mut second));
    assert_ne!(sample, bytes(&mut first));
}

#[test]
fn uname_says_riscv32_linux() {
    let mut g = Guest::new();
    let buf = g.buf(390);
    assert_eq!(g.call(UNAME, &[buf]), 0);
    let fields = g.get(buf, 390);
    assert_eq!(&fields[..6], b"Linux\0");
    assert_eq!(&fields[4 * 65..4 * 65 + 8], b"riscv32\0");
}

#[test]
fn without_a_root_there_are_no_files() {
    let mut g = Guest::without_root();
    assert_eq!(g.open("/anything", O_RDONLY), ENOENT);
    assert_eq!(g.write(1, b"stdio still works"), 17);
}

#[test]
fn pipes_block_when_empty_and_read_eof_once_closed() {
    let mut g = Guest::new();
    let (read, write) = g.pipe(0);
    assert_eq!((read, write), (3, 4));
    let buf = g.buf(16);
    assert_eq!(g.action(READ, &[read, buf, 16]), Action::Block);
    assert_eq!(g.write(write as i32, b"data"), 4);
    assert_eq!(g.read(read as i32, 16), b"data");
    assert_eq!(g.call(CLOSE, &[write]), 0);
    assert_eq!(g.call(READ, &[read, buf, 16]), 0, "end of file");
}

#[test]
fn nonblocking_pipes_return_eagain() {
    let mut g = Guest::new();
    let (read, _write) = g.pipe(O_NONBLOCK);
    let buf = g.buf(16);
    assert_eq!(g.call(READ, &[read, buf, 16]), EAGAIN);
}

#[test]
fn writing_to_an_unread_pipe_raises_sigpipe_unless_ignored() {
    let mut g = Guest::new();
    let (read, write) = g.pipe(0);
    g.call(CLOSE, &[read]);
    let data = g.put(b"x");
    let killed = Action::Exit(Termination::Killed(SIGPIPE));
    assert_eq!(g.action(WRITE, &[write, data, 1]), killed);

    let sig_ign = g.put(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(g.call(RT_SIGACTION, &[SIGPIPE, sig_ign, 0, 8]), 0);
    assert_eq!(g.write(write as i32, b"x"), EPIPE);
}

#[test]
fn process_calls_become_container_actions() {
    let mut g = Guest::new();
    let fork = Action::Fork {
        stack: 0,
        parent_tid: 0,
    };
    assert_eq!(g.action(CLONE, &[SIGCHLD, 0, 0, 0, 0]), fork);
    // A thread: its stack, its TLS, and where its id goes
    const CLONE_SETTLS: u32 = 0x80000;
    const CLONE_PARENT_SETTID: u32 = 0x100000;
    const CLONE_CHILD_CLEARTID: u32 = 0x200000;
    let flags = CLONE_VM | CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID;
    assert_eq!(
        g.action(CLONE, &[flags, 0x9000, 0x8100, 0x7000, 0x8100]),
        Action::Thread {
            stack: 0x9000,
            tls: Some(0x7000),
            parent_tid: 0x8100,
            child_tid: 0x8100,
            clear_child_tid: true,
        }
    );
    assert_eq!(
        g.call(CLONE, &[CLONE_THREAD, 0, 0, 0, 0]),
        EINVAL,
        "a thread shares memory"
    );
    assert_eq!(
        g.action(EXIT, &[4]),
        Action::ExitThread(Termination::Exited(4))
    );

    let status = g.buf(4);
    let wait = Wait {
        pid: None,
        nohang: true,
        report: Report::Status(status),
    };
    assert_eq!(
        g.action(WAIT4, &[-1i32 as u32, status, WNOHANG, 0]),
        Action::Wait(wait)
    );
    assert_eq!(
        g.action(KILL, &[5, 15]),
        Action::Kill { pid: 5, signal: 15 }
    );
    assert_eq!(
        g.action(EXIT_GROUP, &[3]),
        Action::Exit(Termination::Exited(3))
    );
}

#[test]
fn execve_reads_the_program_and_follows_shebang_lines() {
    use std::os::unix::fs::PermissionsExt;
    let mut g = Guest::new();
    let program = common::elf(&[common::ECALL], &[]);
    fs::create_dir(g.host("bin")).unwrap();
    for (name, contents) in [
        ("prog", program.clone()),
        ("script", b"#!/bin/prog -x\necho ignored\n".to_vec()),
        ("junk", b"not a program".to_vec()),
        ("plain", program.clone()),
    ] {
        fs::write(g.host(&format!("bin/{name}")), contents).unwrap();
        let mode = if name == "plain" { 0o644 } else { 0o755 };
        fs::set_permissions(
            g.host(&format!("bin/{name}")),
            fs::Permissions::from_mode(mode),
        )
        .unwrap();
    }

    let (script, argv) = (g.cstr("/bin/script"), g.strings(&["script", "arg1"]));
    let envp = g.strings(&["HOME=/"]);
    let expected = Exec {
        path: b"/bin/prog".to_vec(), // the interpreter runs
        elf: program,
        argv: ["/bin/prog", "-x", "/bin/script", "arg1"]
            .map(|a| a.as_bytes().to_vec())
            .to_vec(),
        envp: vec![b"HOME=/".to_vec()],
    };
    assert_eq!(
        g.action(EXECVE, &[script, argv, envp]),
        Action::Exec(expected)
    );

    let (junk, plain) = (g.cstr("/bin/junk"), g.cstr("/bin/plain"));
    assert_eq!(g.call(EXECVE, &[junk, argv, 0]), ENOEXEC);
    assert_eq!(g.call(EXECVE, &[plain, argv, 0]), EACCES);
}
