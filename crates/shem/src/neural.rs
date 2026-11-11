//! A vCPU run by the neural executor (`neural/`), reached over a Unix socket.
//!
//! Each [`Vcpu`] call is a request; cloning asks the server to clone vCPU state.
//! See `docs/neural.md` for the protocol.
//!
//! The `Vcpu` trait has no room for transport errors, so a broken connection to
//! the server panics with the reason.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::vcpu::{Exit, ExitReason, MachineState, MemoryFault, SnapshotError, Vcpu};

const CREATE: u8 = 1;
const DROP: u8 = 2;
const CLONE: u8 = 3;
const RUN: u8 = 4;
const GET_REG: u8 = 5;
const SET_REG: u8 = 6;
const GET_PC: u8 = 7;
const SET_PC: u8 = 8;
const READ_MEM: u8 = 9;
const WRITE_MEM: u8 = 10;
const RESET: u8 = 11;
const SNAPSHOT: u8 = 12;
const RESTORE: u8 = 13;

const OK: u8 = 0;
const MEMORY_FAULT: u8 = 2;

pub struct NeuralVcpu {
    conn: Arc<Mutex<UnixStream>>,
    id: u32,
    mem_size: u32,
}

/// A request the server answered with an error.
#[derive(Debug)]
struct Refused {
    status: u8,
    message: String,
}

impl NeuralVcpu {
    /// Create a vCPU with `mem_size` bytes of memory on the server at `socket`.
    pub fn connect(socket: &Path, mem_size: u32) -> io::Result<Self> {
        Self::connect_with_base(socket, mem_size, 0)
    }

    /// Create a vCPU whose memory occupies `base .. base + mem_size`, as RAM
    /// does in VM mode.
    pub fn connect_with_base(socket: &Path, mem_size: u32, base: u32) -> io::Result<Self> {
        Self::connect_with(socket, mem_size, base, None)
    }

    /// Create a vCPU as `connect_with_base` does, whose CPU is the compiled
    /// layer at `weights` (an image's `model/compiled/rv32im`) rather than
    /// the server's own.
    pub fn connect_with(
        socket: &Path,
        mem_size: u32,
        base: u32,
        weights: Option<&Path>,
    ) -> io::Result<Self> {
        let conn = Arc::new(Mutex::new(UnixStream::connect(socket)?));
        let mut args = [mem_size.to_le_bytes(), base.to_le_bytes()].concat();
        if let Some(weights) = weights {
            args.extend(weights.as_os_str().as_encoded_bytes());
        }
        let reply =
            request(&conn, CREATE, &args)?.map_err(|refused| io::Error::other(refused.message))?;
        Ok(Self {
            conn,
            id: u32::from_le_bytes(reply[..4].try_into().unwrap()),
            mem_size,
        })
    }

    fn call(&self, op: u8, args: &[u8]) -> Result<Vec<u8>, Refused> {
        let payload = [&self.id.to_le_bytes()[..], args].concat();
        match request(&self.conn, op, &payload) {
            Ok(reply) => reply,
            Err(err) => panic!("lost the neural vCPU server: {err}"),
        }
    }

    fn expect(&self, op: u8, args: &[u8]) -> Vec<u8> {
        self.call(op, args).unwrap_or_else(|refused| {
            panic!("neural vCPU server refused op {op}: {}", refused.message)
        })
    }

    fn word(&self, op: u8, args: &[u8]) -> u32 {
        u32::from_le_bytes(self.expect(op, args)[..4].try_into().unwrap())
    }
}

fn request(conn: &Mutex<UnixStream>, op: u8, args: &[u8]) -> io::Result<Result<Vec<u8>, Refused>> {
    let mut stream = conn.lock().unwrap();
    let length = (1 + args.len()) as u32;
    stream.write_all(&[&length.to_le_bytes()[..], &[op], args].concat())?;
    let mut header = [0u8; 4];
    stream.read_exact(&mut header)?;
    let mut reply = vec![0u8; u32::from_le_bytes(header) as usize];
    stream.read_exact(&mut reply)?;
    let (status, body) = reply.split_first().ok_or(io::ErrorKind::UnexpectedEof)?;
    Ok(match *status {
        OK => Ok(body.to_vec()),
        status => Err(Refused {
            status,
            message: String::from_utf8_lossy(body).into_owned(),
        }),
    })
}

impl Vcpu for NeuralVcpu {
    fn pc(&self) -> u32 {
        self.word(GET_PC, &[])
    }

    fn set_pc(&mut self, pc: u32) {
        self.expect(SET_PC, &pc.to_le_bytes());
    }

    fn reg(&self, index: usize) -> u32 {
        self.word(GET_REG, &[index as u8])
    }

    fn set_reg(&mut self, index: usize, value: u32) {
        self.expect(
            SET_REG,
            &[&[index as u8][..], &value.to_le_bytes()].concat(),
        );
    }

    fn mem_size(&self) -> u32 {
        self.mem_size
    }

    fn read_mem(&self, addr: u32, buf: &mut [u8]) -> Result<(), MemoryFault> {
        let args = [addr.to_le_bytes(), (buf.len() as u32).to_le_bytes()].concat();
        match self.call(READ_MEM, &args) {
            Ok(bytes) => {
                buf.copy_from_slice(&bytes);
                Ok(())
            }
            Err(Refused {
                status: MEMORY_FAULT,
                ..
            }) => Err(MemoryFault {
                addr,
                len: buf.len(),
            }),
            Err(refused) => panic!("neural vCPU server refused a read: {}", refused.message),
        }
    }

    fn write_mem(&mut self, addr: u32, data: &[u8]) -> Result<(), MemoryFault> {
        match self.call(WRITE_MEM, &[&addr.to_le_bytes()[..], data].concat()) {
            Ok(_) => Ok(()),
            Err(Refused {
                status: MEMORY_FAULT,
                ..
            }) => Err(MemoryFault {
                addr,
                len: data.len(),
            }),
            Err(refused) => panic!("neural vCPU server refused a write: {}", refused.message),
        }
    }

    fn reset(&mut self) {
        self.expect(RESET, &[]);
    }

    fn run(&mut self, budget: u64) -> Exit {
        let reply = self.expect(RUN, &budget.to_le_bytes());
        let word = |at: usize| u32::from_le_bytes(reply[at..at + 4].try_into().unwrap());
        let info = word(1);
        let reason = match reply[0] {
            0 => ExitReason::Budget,
            1 => ExitReason::Ecall,
            2 => ExitReason::Ebreak,
            3 => ExitReason::IllegalInstruction(info),
            4 => ExitReason::MemoryFault(info),
            other => panic!("neural vCPU server sent unknown exit reason {other}"),
        };
        Exit {
            reason,
            pc: word(5),
            steps: u64::from_le_bytes(reply[9..17].try_into().unwrap()),
        }
    }

    fn snapshot(&self) -> MachineState {
        MachineState::decode(&self.expect(SNAPSHOT, &[]))
            .expect("neural vCPU server sent a malformed snapshot")
    }

    fn restore(&mut self, state: &MachineState) -> Result<(), SnapshotError> {
        if state.mem.len() != self.mem_size as usize {
            return Err(SnapshotError::MemorySize {
                expected: self.mem_size as usize,
                found: state.mem.len(),
            });
        }
        self.expect(RESTORE, &state.encode());
        Ok(())
    }
}

impl Clone for NeuralVcpu {
    /// Copy the vCPU on the server, sharing the connection.
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
            id: self.word(CLONE, &[]),
            mem_size: self.mem_size,
        }
    }
}

impl Drop for NeuralVcpu {
    fn drop(&mut self) {
        // The server may already be gone; there is nothing useful to do then.
        let payload = self.id.to_le_bytes();
        let _ = request(&self.conn, DROP, &payload);
    }
}
