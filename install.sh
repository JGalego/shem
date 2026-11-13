#!/bin/sh
# Install shem from a GitHub release:
#
#   curl -fsSL https://jgalego.github.io/shem/install.sh | sh
#
# It installs the shem binary and, on Linux, the containerd shim that Docker
# and containerd find by name, checks the download against its SHA-256, and
# installs uv, which model images need, when it is missing. Settings, as
# environment variables (pass them to sh: `curl ... | SHEM_VERSION=0.2.0 sh`):
#
#   SHEM_VERSION         release to install, such as 0.2.0 (default: the latest)
#   SHEM_INSTALL_DIR     where the binaries go (default: /usr/local/bin, so
#                        containerd finds the shim; uses sudo when needed)
#   SHEM_SKIP_UV         set to 1 to leave uv alone
#   SHEM_NEURAL_SERVICE  set to 1 to run the model server as a systemd service,
#                        for model images under Docker (Linux only)
#
# shem-uninstall.sh, next to the binaries, removes what this installed.
set -eu

repo=JGalego/shem
install_dir=${SHEM_INSTALL_DIR:-/usr/local/bin}
socket=/tmp/shem-neural.sock

info() { printf '\033[1m[shem]\033[0m %s\n' "$*"; }
fatal() { printf '\033[1;31m[shem]\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || fatal "$1 is required"; }
need curl
need tar
need uname

# The release's binaries are built for these targets
os=$(uname -s) arch=$(uname -m)
case "$os/$arch" in
    Linux/x86_64 | Linux/amd64) target=x86_64-unknown-linux-gnu ;;
    Linux/aarch64 | Linux/arm64) target=aarch64-unknown-linux-gnu ;;
    Darwin/arm64) target=aarch64-apple-darwin ;;
    *) fatal "no release binary for $os $arch; build from source: https://github.com/$repo#getting-started" ;;
esac

# The latest release's tag, from where /releases/latest redirects, which
# needs no API token and has no rate limit
version=${SHEM_VERSION:-}
if [ -z "$version" ]; then
    latest=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$repo/releases/latest")
    version=${latest##*/v}
    case "$version" in
        [0-9]*) ;;
        *) fatal "could not find the latest release" ;;
    esac
fi
version=${version#v}

# Write to install_dir directly, or through sudo when it is not writable
sudo=
if ! mkdir -p "$install_dir" 2>/dev/null || [ ! -w "$install_dir" ]; then
    command -v sudo >/dev/null 2>&1 || fatal "$install_dir is not writable and there is no sudo; set SHEM_INSTALL_DIR"
    sudo=sudo
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

name="shem-$version-$target"
url="https://github.com/$repo/releases/download/v$version/$name.tar.gz"
info "Downloading shem $version for $target"
curl -fsSL -o "$tmp/$name.tar.gz" "$url" || fatal "no release at $url"
curl -fsSL -o "$tmp/$name.tar.gz.sha256" "$url.sha256" || fatal "no checksum at $url.sha256"

expected=$(cut -d' ' -f1 "$tmp/$name.tar.gz.sha256")
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp/$name.tar.gz" | cut -d' ' -f1)
else
    actual=$(shasum -a 256 "$tmp/$name.tar.gz" | cut -d' ' -f1)
fi
[ "$expected" = "$actual" ] || fatal "checksum mismatch for $name.tar.gz: expected $expected, got $actual"

tar xzf "$tmp/$name.tar.gz" -C "$tmp"
installed="shem"

$sudo install -m 755 "$tmp/$name/shem" "$install_dir/shem"
if [ -f "$tmp/$name/containerd-shim-shem-v1" ]; then
    $sudo install -m 755 "$tmp/$name/containerd-shim-shem-v1" "$install_dir/containerd-shim-shem-v1"
    installed="$installed containerd-shim-shem-v1"
fi
info "Installed $installed in $install_dir"

# Model images import and run models with uv, from the matching shem-neural
# release on PyPI
if [ "${SHEM_SKIP_UV:-0}" != 1 ] && ! command -v uv >/dev/null 2>&1; then
    info "Installing uv, which model images need"
    curl -fsSL https://astral.sh/uv/install.sh | sh -s -- --quiet
fi

# The model server for model containers under Docker, which runs as root
# because it reads the root filesystems containerd mounts
service=
if [ "${SHEM_NEURAL_SERVICE:-0}" = 1 ]; then
    [ "$os" = Linux ] || fatal "SHEM_NEURAL_SERVICE needs Linux"
    command -v systemctl >/dev/null 2>&1 || fatal "SHEM_NEURAL_SERVICE needs systemd"
    uv=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
    [ -x "$uv" ] || fatal "SHEM_NEURAL_SERVICE needs uv"
    command -v sudo >/dev/null 2>&1 || [ "$(id -u)" = 0 ] || fatal "SHEM_NEURAL_SERVICE needs root"
    root=; [ "$(id -u)" = 0 ] || root=sudo
    $root install -m 755 "$uv" /usr/local/bin/uv 2>/dev/null || true
    $root tee /etc/systemd/system/shem-neural.service >/dev/null <<EOF
[Unit]
Description=shem model server
After=network-online.target

[Service]
ExecStart=/usr/local/bin/uv tool run --from shem-neural==$version shem-neural-server --socket $socket
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
    $root systemctl daemon-reload
    $root systemctl enable --now shem-neural.service
    service=shem-neural.service
    info "Started $service, listening at $socket"
fi

# What this installed, to remove it again
{
    echo '#!/bin/sh'
    echo '# Remove what install.sh installed. Images stay in ~/.local/share/shem.'
    echo 'set -eu'
    if [ -n "$service" ]; then
        echo "sudo systemctl disable --now $service || true"
        echo "sudo rm -f /etc/systemd/system/$service"
        echo 'sudo systemctl daemon-reload'
    fi
    echo "cd '$install_dir'"
    echo "${sudo:+$sudo }rm -f $installed shem-uninstall.sh"
    echo "echo 'Removed shem from $install_dir'"
} > "$tmp/shem-uninstall.sh"
$sudo install -m 755 "$tmp/shem-uninstall.sh" "$install_dir/shem-uninstall.sh"

case ":$PATH:" in
    *":$install_dir:"*) ;;
    *) info "Add $install_dir to your PATH" ;;
esac
command -v oras >/dev/null 2>&1 || info "Pulling and pushing images needs ORAS: https://oras.land/docs/installation"
# Docker Desktop, and Docker on macOS, run containerd in a VM, which cannot
# see the shim: Docker builds images there, and shem runs them
if [ "$os" = Darwin ] || docker info --format '{{.OperatingSystem}}' 2>/dev/null | grep -q 'Docker Desktop'; then
    info "Docker runs containers in a VM here, where the shim is not. Run the images docker build makes with: docker save IMAGE | shem load && shem run IMAGE"
fi
info "Done. Try: shem run ghcr.io/jgalego/shem/hello:latest"
