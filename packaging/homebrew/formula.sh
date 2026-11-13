#!/bin/sh
# Write the Homebrew formula for a release, from its binaries' checksums:
#   packaging/homebrew/formula.sh VERSION DIR > shem.rb
# DIR holds the release's shem-VERSION-TARGET.tar.gz.sha256 files.
set -eu
version=$1 dir=$2
url="https://github.com/JGalego/shem/releases/download/v$version"
sha() { cut -d' ' -f1 "$dir/shem-$version-$1.tar.gz.sha256"; }
cat <<FORMULA
class Shem < Formula
  desc "Run containers and VMs inside a model's weights"
  homepage "https://github.com/JGalego/shem"
  version "$version"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "$url/shem-$version-aarch64-apple-darwin.tar.gz"
      sha256 "$(sha aarch64-apple-darwin)"
    end
  end

  on_linux do
    on_intel do
      url "$url/shem-$version-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "$(sha x86_64-unknown-linux-gnu)"
    end
    on_arm do
      url "$url/shem-$version-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "$(sha aarch64-unknown-linux-gnu)"
    end
  end

  # Model images and the neural vCPU run on shem-neural, which uv fetches
  depends_on "uv"

  def install
    bin.install "shem"
    bin.install "containerd-shim-shem-v1" if OS.linux?
  end

  test do
    assert_match "shem #{version}", shell_output("#{bin}/shem --version")
  end
end
FORMULA
