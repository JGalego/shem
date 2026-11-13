//! Programs on terminals of their own, as `docker run -t` gives them. A key
//! that sends a signal goes to whichever program the shim's process runs,
//! so these run one after another, in a process apart from other tests.
#![cfg(target_os = "linux")]

mod common;

use std::fs::File;
use std::io::{PipeWriter, Write};
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::*;
use containerd_shim_shem::bundle::{self, Bundle, Control};
use containerd_shim_shem::console::Console;

/// A program running on a terminal of its own.
struct OnTerminal {
    _dir: tempfile::TempDir,
    keys: PipeWriter,
    shown: PathBuf,
    run: JoinHandle<u8>,
    console: Console,
}

impl OnTerminal {
    /// Start `program` on a terminal of `size` (rows, columns), which shows
    /// what it gets in a file, as containerd's stdout FIFO would take it.
    fn start(program: Vec<u8>, size: Option<(u16, u16)>) -> Self {
        let dir = bundle_with(program);
        let bundle = Bundle::load(dir.path()).unwrap();
        let control = Arc::new(Control::default());
        let (typed, keys) = std::io::pipe().unwrap();
        let shown = dir.path().join("stdout");
        File::create(&shown).unwrap();
        let typed = File::from(OwnedFd::from(typed));
        let (console, stdio) =
            Console::open(Some(typed), shown.to_str().unwrap(), control.clone()).unwrap();
        if let Some((rows, columns)) = size {
            console.resizer().unwrap().resize(rows, columns);
        }
        let terminal = console.fd();
        let run = std::thread::spawn(move || {
            bundle::run(&bundle, stdio, &control, Some(terminal)).unwrap()
        });
        Self {
            _dir: dir,
            keys,
            shown,
            run,
            console,
        }
    }

    fn wait_for(&self, text: &[u8]) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::fs::read(&self.shown).unwrap() != text {
            assert!(
                Instant::now() < deadline,
                "the terminal never showed {text:?}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn type_keys(&mut self, keys: &[u8]) {
        self.keys.write_all(keys).unwrap();
    }

    /// The program's status, and everything the terminal showed.
    fn end(self) -> (u8, Vec<u8>) {
        let status = self.run.join().unwrap();
        self.console.finish();
        (status, std::fs::read(&self.shown).unwrap())
    }
}

/// A program that asks for its terminal's window size, writes it, and exits.
fn window_size() -> Vec<u8> {
    let buf = BASE + 84;
    let mut code = vec![addi(10, 0, 1)]; // ioctl(1, TIOCGWINSZ, buf)
    code.extend(li(11, 0x5413));
    code.extend(li(12, buf));
    code.extend([addi(17, 0, 29), ECALL]);
    code.push(addi(10, 0, 1)); // write(1, buf, 8)
    code.extend(li(11, buf));
    code.extend([addi(12, 0, 8), addi(17, 0, 64), ECALL]);
    code.extend([addi(10, 0, 0), addi(17, 0, 93), ECALL]); // exit(0)
    elf(&code, &[0; 8])
}

#[test]
fn programs_get_terminals_of_their_own() {
    // The terminal echoes what is typed, and ends lines with "\r\n"
    let mut echo_ = OnTerminal::start(echo(), None);
    echo_.wait_for(b"ready\r\n");
    echo_.type_keys(b"ping\n");
    assert_eq!(echo_.end(), (0, b"ready\r\nping\r\nping\r\n".to_vec()));

    // Ctrl-C interrupts the program as it waits for input
    let mut interrupted = OnTerminal::start(echo(), None);
    interrupted.wait_for(b"ready\r\n");
    std::thread::sleep(Duration::from_millis(200)); // well into its read
    interrupted.type_keys(b"\x03");
    assert_eq!(interrupted.end().0, 128 + 2);

    // Ctrl-D ends its input
    let mut ended = OnTerminal::start(echo(), None);
    ended.wait_for(b"ready\r\n");
    ended.type_keys(b"\x04");
    assert_eq!(ended.end(), (0, b"ready\r\n".to_vec()));

    // Its window is the size docker run gives it
    let sized = OnTerminal::start(window_size(), Some((24, 80)));
    assert_eq!(sized.end(), (0, vec![24, 0, 80, 0, 0, 0, 0, 0]));
}
