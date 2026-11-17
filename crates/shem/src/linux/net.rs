//! Stream sockets among the container's own processes.
//!
//! A container has a network of its own and nothing else: processes can
//! listen on and connect to `AF_UNIX` paths and `AF_INET` ports on loopback,
//! and talk through them, but no socket leaves the container, so a guest that
//! is untrusted can serve and call its siblings, a Pod's containers' worth of
//! programs on one vCPU or many, and reach no host. A connection is two pipes,
//! one each way, so a socket reads, writes, blocks, polls and closes as a pipe
//! does, and a connection is deterministic as a pipe is.
//!
//! Not here: datagrams, `AF_INET6`, addresses other than loopback, passing
//! descriptors, and a path in the filesystem for a Unix socket (its name lives
//! in the container's network, not on its disk).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use super::abi::*;
use super::files::{Description, Pipe, PipeEnd};
use super::{Linux, Process};
use crate::vcpu::Vcpu;

const AF_UNIX: u32 = 1;
const AF_INET: u32 = 2;
const SOCK_STREAM: u32 = 1;
const SOCK_TYPE_MASK: u32 = 0xf;
const SOCK_NONBLOCK: u32 = 0o4000;
const SOCK_CLOEXEC: u32 = 0o2000000;
const MSG_DONTWAIT: u32 = 0x40;
const SHUT_RD: u32 = 0;
const SHUT_WR: u32 = 1;
const SHUT_RDWR: u32 = 2;
const SOL_SOCKET: u32 = 1;
const SO_TYPE: u32 = 3;
const SO_ERROR: u32 = 4;
const SO_ACCEPTCONN: u32 = 30;
const O_NONBLOCK: u32 = 0o4000;
/// `sockaddr_un`'s path holds this many bytes
const UNIX_PATH_MAX: usize = 108;
/// Connections a listener holds that no one has accepted yet, at most
const MAX_BACKLOG: usize = 128;
/// Ports `bind` to port 0 hands out, from the start of Linux's range
const EPHEMERAL: u16 = 32768;

/// A place in the container's network.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Addr {
    Unix(Vec<u8>),
    /// A port on loopback
    Inet(u16),
}

impl Addr {
    /// A `sockaddr` for it, as `getsockname` and `accept` write them.
    fn encode(&self) -> Vec<u8> {
        match self {
            Addr::Unix(path) => {
                let mut out = (AF_UNIX as u16).to_le_bytes().to_vec();
                out.extend(path);
                out.push(0);
                out
            }
            Addr::Inet(port) => {
                let mut out = (AF_INET as u16).to_le_bytes().to_vec();
                out.extend(port.to_be_bytes());
                out.extend([127, 0, 0, 1]);
                out.extend([0; 8]);
                out
            }
        }
    }
}

/// The container's network: who listens where, and which names are taken.
#[derive(Default)]
pub struct Net {
    listeners: HashMap<Addr, Weak<Mutex<Listener>>>,
    taken: HashSet<Addr>,
    next_port: u16,
}

impl Net {
    fn ephemeral(&mut self) -> Result<u16, Errno> {
        for _ in 0..(u16::MAX - EPHEMERAL) {
            let port = EPHEMERAL + self.next_port % (u16::MAX - EPHEMERAL);
            self.next_port = self.next_port.wrapping_add(1);
            if !self.taken.contains(&Addr::Inet(port)) {
                return Ok(port);
            }
        }
        Err(Errno::EADDRINUSE)
    }
}

type Shared = Arc<Mutex<Net>>;

struct Listener {
    queue: VecDeque<Connection>,
    backlog: usize,
}

/// A connection made and not yet accepted: the server's two ends.
struct Connection {
    rx: PipeEnd,
    tx: PipeEnd,
    local: Addr,
    peer: Option<Addr>,
}

enum State {
    Fresh,
    Bound(Addr),
    Listening(Addr, Arc<Mutex<Listener>>),
    Connected {
        /// None once shut down for reading, or writing
        rx: Option<PipeEnd>,
        tx: Option<PipeEnd>,
        local: Option<Addr>,
        peer: Option<Addr>,
        /// Whether `local` is this socket's own name, which closing it frees:
        /// not a server's, which an accepted connection shares with its listener
        owns: bool,
    },
}

pub(super) struct Socket {
    domain: u32,
    state: State,
    net: Shared,
}

impl Drop for Socket {
    fn drop(&mut self) {
        let (addr, listening) = match &self.state {
            State::Bound(addr) => (addr, false),
            State::Listening(addr, _) => (addr, true),
            State::Connected {
                local: Some(addr),
                owns: true,
                ..
            } => (addr, false),
            _ => return,
        };
        let mut net = self.net.lock().unwrap();
        net.taken.remove(addr);
        if listening {
            net.listeners.remove(addr);
        }
    }
}

impl Socket {
    fn new(domain: u32, net: Shared) -> Self {
        Self {
            domain,
            state: State::Fresh,
            net,
        }
    }

    fn connected(domain: u32, net: Shared, c: Connection) -> Self {
        Self {
            domain,
            net,
            state: State::Connected {
                rx: Some(c.rx),
                tx: Some(c.tx),
                local: Some(c.local),
                peer: c.peer,
                owns: false,
            },
        }
    }

    pub(super) fn read(&mut self, buf: &mut [u8], nonblock: bool) -> Result<usize, Errno> {
        match &self.state {
            State::Connected { rx: Some(rx), .. } => rx.read(buf, nonblock),
            State::Connected { rx: None, .. } => Ok(0),
            _ => Err(Errno::ENOTCONN),
        }
    }

    pub(super) fn write(&mut self, data: &[u8], nonblock: bool) -> Result<usize, Errno> {
        match &self.state {
            State::Connected { tx: Some(tx), .. } => tx.write(data, nonblock),
            State::Connected { tx: None, .. } => Err(Errno::EPIPE),
            _ => Err(Errno::EPIPE),
        }
    }

    pub(super) fn ready(&self) -> u16 {
        const POLLIN: u16 = 0x1;
        const POLLOUT: u16 = 0x4;
        const POLLHUP: u16 = 0x10;
        match &self.state {
            State::Listening(_, listener) => {
                if listener.lock().unwrap().queue.is_empty() {
                    0
                } else {
                    POLLIN
                }
            }
            State::Connected { rx, tx, .. } => {
                let readable = rx.as_ref().map_or(POLLIN | POLLHUP, PipeEnd::ready);
                let writable = tx.as_ref().map_or(0, PipeEnd::ready);
                // The write end's POLLERR means the peer is gone, which a socket reads as hang-up too
                (readable & (POLLIN | POLLHUP)) | (writable & POLLOUT)
            }
            _ => 0,
        }
    }
}

/// The address a guest's `sockaddr` names, in the domain it is for.
fn parse(vcpu: &dyn Vcpu, domain: u32, addr: u32, len: u32) -> Result<Addr, Errno> {
    if !(2..=128).contains(&len) {
        return Err(Errno::EINVAL);
    }
    let mut raw = vec![0; len as usize];
    vcpu.read_mem(addr, &mut raw)?;
    if u32::from(u16::from_le_bytes([raw[0], raw[1]])) != domain {
        return Err(Errno::EAFNOSUPPORT);
    }
    match domain {
        AF_UNIX => {
            let path = &raw[2..];
            let path = &path[..path.iter().position(|&b| b == 0).unwrap_or(path.len())];
            if path.len() >= UNIX_PATH_MAX {
                return Err(Errno::EINVAL);
            }
            if path.is_empty() {
                return Err(Errno::EINVAL); // unnamed and abstract names are not here
            }
            Ok(Addr::Unix(path.to_vec()))
        }
        _ => {
            if len < 8 {
                return Err(Errno::EINVAL);
            }
            let ip = &raw[4..8];
            // Loopback, or any address of the container's own, which is loopback
            if ip != [0, 0, 0, 0] && ip[0] != 127 {
                return Err(Errno::EADDRNOTAVAIL);
            }
            Ok(Addr::Inet(u16::from_be_bytes([raw[2], raw[3]])))
        }
    }
}

/// Write `addr` to the guest's `sockaddr` buffer of `*len` bytes, setting
/// `*len` to the address's length, as `getsockname` and `accept` do.
fn put(vcpu: &mut dyn Vcpu, addr: Option<&Addr>, buf: u32, len: u32) -> Result<(), Errno> {
    if buf == 0 || len == 0 {
        return Ok(());
    }
    let mut size = [0; 4];
    vcpu.read_mem(len, &mut size)?;
    let room = u32::from_le_bytes(size) as usize;
    let bytes = addr.map_or_else(|| (AF_UNIX as u16).to_le_bytes().to_vec(), Addr::encode);
    vcpu.write_mem(buf, &bytes[..bytes.len().min(room)])?;
    vcpu.write_mem(len, &(bytes.len() as u32).to_le_bytes())?;
    Ok(())
}

fn nonblocking(description: &Description) -> bool {
    description.flags() & O_NONBLOCK != 0
}

impl Linux {
    pub(super) fn socket(
        &mut self,
        p: &mut Process,
        domain: u32,
        kind: u32,
        protocol: u32,
    ) -> Result<u32, Errno> {
        if domain != AF_UNIX && domain != AF_INET {
            return Err(Errno::EAFNOSUPPORT);
        }
        if kind & SOCK_TYPE_MASK != SOCK_STREAM {
            return Err(Errno::ESOCKTNOSUPPORT);
        }
        if kind & !(SOCK_TYPE_MASK | SOCK_NONBLOCK | SOCK_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        if protocol != 0 && !(domain == AF_INET && protocol == 6) {
            return Err(Errno::EPROTONOSUPPORT);
        }
        let socket = Socket::new(domain, self.net.clone());
        p.install_socket(socket, kind & SOCK_NONBLOCK, kind & SOCK_CLOEXEC != 0)
    }

    pub(super) fn socketpair(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        domain: u32,
        kind: u32,
        protocol: u32,
        fds: u32,
    ) -> Result<u32, Errno> {
        if domain != AF_UNIX {
            return Err(Errno::EOPNOTSUPP);
        }
        if kind & SOCK_TYPE_MASK != SOCK_STREAM {
            return Err(Errno::ESOCKTNOSUPPORT);
        }
        if protocol != 0 {
            return Err(Errno::EPROTONOSUPPORT);
        }
        let (one, other) = (Pipe::shared(), Pipe::shared());
        let end = |pipe: &Arc<Mutex<Pipe>>, writer| PipeEnd {
            pipe: pipe.clone(),
            writer,
        };
        let make = |rx, tx, net: &Shared| Socket {
            domain,
            net: net.clone(),
            state: State::Connected {
                rx: Some(rx),
                tx: Some(tx),
                local: None,
                peer: None,
                owns: false,
            },
        };
        let a = make(end(&other, false), end(&one, true), &self.net);
        let b = make(end(&one, false), end(&other, true), &self.net);
        let cloexec = kind & SOCK_CLOEXEC != 0;
        let first = p.install_socket(a, kind & SOCK_NONBLOCK, cloexec)?;
        let installed = p
            .install_socket(b, kind & SOCK_NONBLOCK, cloexec)
            .and_then(|second| {
                let pair = [first.to_le_bytes(), second.to_le_bytes()].concat();
                vcpu.write_mem(fds, &pair).map_err(|fault| {
                    p.close(second).ok();
                    Errno::from(fault)
                })
            });
        if let Err(errno) = installed {
            p.close(first).ok();
            return Err(errno);
        }
        Ok(0)
    }

    pub(super) fn bind(
        &mut self,
        p: &mut Process,
        vcpu: &dyn Vcpu,
        fd: u32,
        addr: u32,
        len: u32,
    ) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        if !matches!(socket.state, State::Fresh) {
            return Err(Errno::EINVAL);
        }
        let mut addr = parse(vcpu, socket.domain, addr, len)?;
        let mut net = self.net.lock().unwrap();
        if let Addr::Inet(0) = addr {
            addr = Addr::Inet(net.ephemeral()?);
        }
        if !net.taken.insert(addr.clone()) {
            return Err(Errno::EADDRINUSE);
        }
        socket.state = State::Bound(addr);
        Ok(0)
    }

    pub(super) fn listen(&mut self, p: &mut Process, fd: u32, backlog: u32) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        let backlog = (backlog as i32).clamp(1, MAX_BACKLOG as i32) as usize;
        let addr = match &socket.state {
            State::Bound(addr) => addr.clone(),
            State::Listening(_, listener) => {
                listener.lock().unwrap().backlog = backlog; // as Linux lets a listener's change
                return Ok(0);
            }
            _ => return Err(Errno::EINVAL),
        };
        let listener = Arc::new(Mutex::new(Listener {
            queue: VecDeque::new(),
            backlog,
        }));
        self.net
            .lock()
            .unwrap()
            .listeners
            .insert(addr.clone(), Arc::downgrade(&listener));
        socket.state = State::Listening(addr, listener);
        Ok(0)
    }

    pub(super) fn accept(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        addr: u32,
        len: u32,
        flags: u32,
    ) -> Result<u32, Errno> {
        if flags & !(SOCK_NONBLOCK | SOCK_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        let description = p.socket_of(fd)?;
        let (connection, domain) = {
            let mut description = description.lock().unwrap();
            let nonblock = nonblocking(&description);
            let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
            let State::Listening(_, listener) = &socket.state else {
                return Err(Errno::EINVAL);
            };
            let connection = listener.lock().unwrap().queue.pop_front();
            match connection {
                Some(connection) => (connection, socket.domain),
                None if nonblock => return Err(Errno::EAGAIN),
                None => return Err(Errno::WOULD_BLOCK),
            }
        };
        put(vcpu, connection.peer.as_ref(), addr, len)?;
        let socket = Socket::connected(domain, self.net.clone(), connection);
        p.install_socket(socket, flags & SOCK_NONBLOCK, flags & SOCK_CLOEXEC != 0)
    }

    pub(super) fn connect(
        &mut self,
        p: &mut Process,
        vcpu: &dyn Vcpu,
        fd: u32,
        addr: u32,
        len: u32,
    ) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        let target = parse(vcpu, socket.domain, addr, len)?;
        let own = match &socket.state {
            State::Fresh => None,
            State::Bound(addr) => Some(addr.clone()),
            State::Connected { .. } => return Err(Errno::EISCONN),
            State::Listening(..) => return Err(Errno::EINVAL),
        };
        let mut net = self.net.lock().unwrap();
        let listener = net
            .listeners
            .get(&target)
            .and_then(Weak::upgrade)
            .ok_or(Errno::ECONNREFUSED)?;
        let mut listener = listener.lock().unwrap();
        if listener.queue.len() >= listener.backlog {
            return Err(Errno::ECONNREFUSED);
        }
        // A client of an inet server has a port of its own, as the server sees it
        let (local, owns) = match (own, socket.domain) {
            (Some(addr), _) => (Some(addr), true),
            (None, AF_INET) => {
                let port = Addr::Inet(net.ephemeral()?);
                net.taken.insert(port.clone());
                (Some(port), true)
            }
            (None, _) => (None, false),
        };
        let (up, down) = (Pipe::shared(), Pipe::shared());
        let end = |pipe: &Arc<Mutex<Pipe>>, writer| PipeEnd {
            pipe: pipe.clone(),
            writer,
        };
        listener.queue.push_back(Connection {
            rx: end(&up, false),
            tx: end(&down, true),
            local: target.clone(),
            peer: local.clone(),
        });
        socket.state = State::Connected {
            rx: Some(end(&down, false)),
            tx: Some(end(&up, true)),
            local,
            peer: Some(target),
            owns,
        };
        Ok(0)
    }

    pub(super) fn sockname(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        addr: u32,
        len: u32,
        peer: bool,
    ) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        let found = match (&socket.state, peer) {
            (State::Bound(addr) | State::Listening(addr, _), false) => Some(addr.clone()),
            (State::Connected { local, .. }, false) => local.clone(),
            (State::Connected { peer, .. }, true) => peer.clone(),
            (_, true) => return Err(Errno::ENOTCONN),
            (State::Fresh, false) => None,
        };
        put(vcpu, found.as_ref(), addr, len)?;
        Ok(0)
    }

    pub(super) fn shutdown(&mut self, p: &mut Process, fd: u32, how: u32) -> Result<u32, Errno> {
        if how > SHUT_RDWR {
            return Err(Errno::EINVAL);
        }
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        let State::Connected { rx, tx, .. } = &mut socket.state else {
            return Err(Errno::ENOTCONN);
        };
        // Dropping an end is how the peer sees end of file, or EPIPE
        if how == SHUT_RD || how == SHUT_RDWR {
            rx.take();
        }
        if how == SHUT_WR || how == SHUT_RDWR {
            tx.take();
        }
        Ok(0)
    }

    pub(super) fn setsockopt(&mut self, p: &mut Process, fd: u32) -> Result<u32, Errno> {
        p.socket_of(fd)?;
        // Options that tune a real network, such as SO_REUSEADDR and TCP_NODELAY, change nothing here
        Ok(0)
    }

    /// `getsockopt(fd, level, name, value, len)`, in `a`.
    pub(super) fn getsockopt(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        a: &[u32],
    ) -> Result<u32, Errno> {
        let [fd, level, name, value, len] = a[..5] else {
            return Err(Errno::EINVAL);
        };
        let description = p.socket_of(fd)?;
        let mut description = description.lock().unwrap();
        let socket = description.socket_mut().ok_or(Errno::ENOTSOCK)?;
        let answer = match (level, name) {
            (SOL_SOCKET, SO_TYPE) => SOCK_STREAM,
            (SOL_SOCKET, SO_ACCEPTCONN) => u32::from(matches!(socket.state, State::Listening(..))),
            (SOL_SOCKET, SO_ERROR) => 0,
            _ => 0,
        };
        let mut size = [0; 4];
        vcpu.read_mem(len, &mut size)?;
        let room = (u32::from_le_bytes(size) as usize).min(4);
        vcpu.write_mem(value, &answer.to_le_bytes()[..room])?;
        vcpu.write_mem(len, &(room as u32).to_le_bytes())?;
        Ok(0)
    }

    /// `sendto` and `recvfrom` on a stream socket are `write` and `read`
    /// with a flag that may say not to wait.
    pub(super) fn sendto(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        buf: u32,
        count: u32,
        flags: u32,
    ) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        description
            .lock()
            .unwrap()
            .socket_mut()
            .ok_or(Errno::ENOTSOCK)?;
        let restore = dontwait(&description, flags);
        let result = p.write(vcpu, fd, buf, count);
        restore();
        result
    }

    pub(super) fn recvfrom(
        &mut self,
        p: &mut Process,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        buf: u32,
        count: u32,
        flags: u32,
    ) -> Result<u32, Errno> {
        let description = p.socket_of(fd)?;
        description
            .lock()
            .unwrap()
            .socket_mut()
            .ok_or(Errno::ENOTSOCK)?;
        let restore = dontwait(&description, flags);
        let result = p.read(vcpu, fd, buf, count);
        restore();
        result
    }
}

/// Make the descriptor non-blocking for one call, if `flags` says so, and
/// return what puts it back.
fn dontwait(description: &Arc<Mutex<Description>>, flags: u32) -> impl FnOnce() + use<> {
    let before = description.lock().unwrap().flags();
    if flags & MSG_DONTWAIT != 0 {
        description.lock().unwrap().set_flags(before | O_NONBLOCK);
    }
    let description = description.clone();
    move || description.lock().unwrap().set_flags(before)
}
