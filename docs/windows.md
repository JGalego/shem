# shem on Windows

`shem.exe` runs on Windows 10 and later without WSL: the runtime, the image store and the commands that need no Linux kernel. Build it with `cargo install --path crates/shem` (Rust 1.88 or later), or take `shem-VERSION-x86_64-pc-windows-msvc.tar.gz` from a release. Models need [uv](https://docs.astral.sh/uv/) on the `PATH`, as everywhere.

```powershell
shem build -t hello examples/hello
shem run hello
shem run qwen3 "What is a golem?"
```

The store is in `%LOCALAPPDATA%\shem` (or `SHEM_HOME`). Everything else in the docs works the same, with these exceptions.

## What Windows does not have yet

| Left out | Why | What happens |
|---|---|---|
| The model server that outlives `shem run`, and `--vcpu neural` | Both are reached over Unix sockets, which Rust's standard library has not got on Windows | A model run starts the driver itself and loads the weights each time; `--vcpu neural` is refused. |
| Overlays for runs and builds | overlayfs is Linux's | A run unpacks its image into a directory of its own, as it did before overlays; `--root-overlay` warns, and the guest's changes go to the directory. |
| Layers shared between images | Needs the link counts of Unix's file metadata | Each image keeps its own blobs and unpacked files. |
| A raw terminal | Windows' console is not a termios terminal | A guest reads lines from the console; a full-screen program or a shell that edits its own lines does not work. Ctrl-C ends shem. |
| The containerd shim and `docker run --runtime io.containerd.shem.v1` | The shim is Linux's, and Docker Desktop runs containers in a VM | Build with Docker and `shem load`, as [on Docker Desktop](containerd.md#docker-desktop). |
| Special files | Windows has no FIFOs, device nodes or sockets as files | A guest that makes one gets an error. |

## What differs

- A guest's file names must be names Windows can have: one with `\ : * ? " < > |` is refused, and a name that is not UTF-8 cannot be made.
- Files have no permission bits beyond read-only: every file looks executable to the guest, and a layer built on Windows records its files that way, so an image built there is not digest for digest the one built on Linux or macOS.
- Symbolic links need a user the system lets make them: an administrator, or developer mode. Without it, a guest's `symlink` fails.
- Inode numbers and link counts are made up (a file's creation time, and one link), and access times are the host's to move.
- A registry's port in an image name (`localhost:5000/app`) is kept in the store under `localhost%5000`.

The Windows build is checked by CI on every change (`cargo test -p shem` on a Windows runner), with the tests of what it has not got skipped. It has not had the breadth of use the Linux build has.
