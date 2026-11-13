//! Terminals for program containers, as `docker run -t` asks for: a
//! pseudo-terminal, whose line discipline the host's kernel runs (echo, line
//! editing, the end of input on Ctrl-D, and the settings the program makes),
//! copied from and to the FIFOs containerd made for the container. The keys
//! that send signals, while the terminal's settings say they do, signal the
//! program's foreground processes, as a terminal of their own would.

use std::ffi::{CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use shem::container::TERMINAL_SIGNAL;
use shem::linux::Stdio;

use crate::bundle::{Control, Wakeable};

pub struct Console {
    /// The terminal's side for the shim, kept to set the window size.
    master: File,
    /// The program's side, which its terminal requests go to.
    slave: File,
    output: JoinHandle<()>,
}

impl Console {
    /// A terminal for a program, between containerd's `stdin` FIFO, if
    /// there is one, already open, and its `stdout` FIFO; and the program's
    /// standard streams on it, whose reads `control` wakes.
    pub fn open(
        stdin: Option<File>,
        stdout: &str,
        control: Arc<Control>,
    ) -> std::io::Result<(Self, Stdio)> {
        let (master, name) = open_master()?;
        let slave = open_slave(&name, libc::O_RDWR)?;
        // A description of its own, which reads of input make nonblocking
        let input = open_slave(&name, libc::O_RDONLY)?;

        let mut out = match stdout {
            "" => None,
            path => Some(OpenOptions::new().write(true).open(path)?),
        };
        let mut from = master.try_clone()?;
        let output = std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            // Until the program's side is closed, when reading gives EIO
            while let Ok(n @ 1..) = from.read(&mut buf) {
                if let Some(file) = &mut out
                    && file.write_all(&buf[..n]).is_err()
                {
                    out = None;
                }
            }
        });
        if let Some(fifo) = stdin {
            let to = master.try_clone()?;
            std::thread::spawn(move || type_into(fifo, to));
        }
        let stdio = Stdio {
            stdin: Box::new(Wakeable::new(input, control)),
            stdout: Box::new(slave.try_clone()?),
            stderr: Box::new(slave.try_clone()?),
        };
        let console = Self {
            master,
            slave,
            output,
        };
        Ok((console, stdio))
    }

    /// The program's side of the terminal, for its terminal requests.
    pub fn fd(&self) -> RawFd {
        self.slave.as_raw_fd()
    }

    /// A handle on the terminal that sets its window size.
    pub fn resizer(&self) -> std::io::Result<Resizer> {
        Ok(Resizer(self.master.try_clone()?))
    }

    /// Once the program has ended and closed its streams: close the
    /// terminal, and wait until the last of its output has been copied.
    pub fn finish(self) {
        drop(self.slave);
        let _ = self.output.join();
    }
}

/// Sets a terminal's window size, as `docker run -t` does when it starts
/// and when its own terminal's size changes.
pub struct Resizer(File);

impl Resizer {
    /// Set the size, and tell the program with SIGWINCH.
    pub fn resize(&self, rows: u16, cols: u16) {
        let size = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: TIOCSWINSZ reads one winsize
        unsafe { libc::ioctl(self.0.as_raw_fd(), libc::TIOCSWINSZ, &size) };
        TERMINAL_SIGNAL.store(libc::SIGWINCH as u32, Ordering::SeqCst);
    }
}

/// A new pseudo-terminal's master, and the name of its slave.
fn open_master() -> std::io::Result<(File, CString)> {
    // SAFETY: posix_openpt returns a descriptor this owns, which grantpt,
    // unlockpt and ptsname_r take, the last with a buffer of the size given
    unsafe {
        let fd = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY | libc::O_CLOEXEC);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let master = File::from_raw_fd(fd);
        let mut name = [0 as libc::c_char; 128];
        if libc::grantpt(fd) != 0
            || libc::unlockpt(fd) != 0
            || libc::ptsname_r(fd, name.as_mut_ptr(), name.len()) != 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok((master, CStr::from_ptr(name.as_ptr()).to_owned()))
    }
}

/// The slave of a pseudo-terminal, which never becomes the shim's own
/// controlling terminal: the keys typed at it must not signal the shim.
fn open_slave(name: &CStr, mode: libc::c_int) -> std::io::Result<File> {
    let path = std::path::Path::new(std::ffi::OsStr::from_bytes(name.to_bytes()));
    OpenOptions::new()
        .read(mode != libc::O_WRONLY)
        .write(mode != libc::O_RDONLY)
        .custom_flags(libc::O_NOCTTY)
        .open(path)
}

/// Copy what is typed into the terminal. A key that sends a signal, while
/// the terminal's settings say it does, goes to the program as that signal.
fn type_into(mut typed: File, mut master: File) {
    let mut buf = [0u8; 4096];
    while let Ok(n @ 1..) = typed.read(&mut buf) {
        let keys = signal_keys(&master);
        let mut from = 0;
        for (i, byte) in buf[..n].iter().enumerate() {
            if let Some(&(_, signal)) = keys.iter().find(|(key, _)| key == byte) {
                if master.write_all(&buf[from..i]).is_err() {
                    return;
                }
                TERMINAL_SIGNAL.store(signal, Ordering::SeqCst);
                from = i + 1;
            }
        }
        if master.write_all(&buf[from..n]).is_err() {
            return;
        }
    }
}

/// The keys that send signals, with their signals, by the terminal's
/// settings, which the master reads as the slave's: none without ISIG.
fn signal_keys(master: &File) -> Vec<(u8, u32)> {
    // SAFETY: tcgetattr fills a termios
    let mut settings: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(master.as_raw_fd(), &mut settings) } != 0
        || settings.c_lflag & libc::ISIG == 0
    {
        return Vec::new();
    }
    [
        (libc::VINTR, libc::SIGINT),
        (libc::VQUIT, libc::SIGQUIT),
        (libc::VSUSP, libc::SIGTSTP),
    ]
    .into_iter()
    .map(|(index, signal)| (settings.c_cc[index], signal as u32))
    // A disabled key is 0 on Linux
    .filter(|&(key, _)| key != 0)
    .collect()
}
