# Guest toolchain: Zig cross-compiles C into static RV32 Linux executables against musl,
# building musl and compiler-rt for the exact target CPU on first use.
FROM debian:trixie-slim AS toolchain

ARG ZIG_VERSION=0.17.1
ARG ZIG_SHA256_AMD64=3581863e3cbcfd0fce0efc682ce7e9e93da96668a977791d356a26d2a7e9b17a
ARG ZIG_SHA256_ARM64=a1546d5646eb184437918d6635ca6b762a0e889eac09cef720ee7a1ed3fe7795
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl make xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
        amd64) arch=x86_64; sha="$ZIG_SHA256_AMD64" ;; \
        arm64) arch=aarch64; sha="$ZIG_SHA256_ARM64" ;; \
        *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://ziglang.org/download/$ZIG_VERSION/zig-$arch-linux-$ZIG_VERSION.tar.xz" -o /tmp/zig.tar.xz \
    && echo "$sha  /tmp/zig.tar.xz" | sha256sum -c - \
    && mkdir /opt/zig \
    && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
    && rm /tmp/zig.tar.xz

ENV PATH=/opt/zig:$PATH
COPY rv32-cc /usr/local/bin/rv32-cc
WORKDIR /src

# The official RISC-V ISA tests, built for shem's environment in riscv-tests/:
# each test is a static Linux executable that exits 0 on success.
FROM toolchain AS riscv-tests-build
ARG RISCV_TESTS_COMMIT=3df703cd52d5f0d8224e3c50bf03c6cb70efff98
WORKDIR /riscv-tests
RUN curl -fsSL "https://github.com/riscv-software-src/riscv-tests/archive/$RISCV_TESTS_COMMIT.tar.gz" \
    | tar -xz --strip-components=1
COPY riscv-tests/riscv_test.h env/shem/riscv_test.h
# -mno-relax: the tests use gp as their test counter, so the linker must not
# relax addresses against it.
RUN mkdir /out \
    && for suite in rv32ui rv32um rv32ua; do \
        for src in isa/$suite/*.S; do \
            rv32-cc -nostdlib -mno-relax -I env/shem -I isa/macros/scalar \
                -o "/out/$suite-$(basename "$src" .S)" "$src" || exit 1; \
        done; \
    done

FROM scratch AS riscv-tests
COPY --from=riscv-tests-build /out/ /

# BusyBox, statically linked for shem guests, installed as a minimal rootfs.
FROM toolchain AS busybox-build
ARG BUSYBOX_VERSION=1.38.0
ARG BUSYBOX_SHA256=34f9ea6ff8636f2c9241153b9114eefa9e65674a45318ae1ef95bb5f31c53bb2
RUN apt-get update \
    && apt-get install -y --no-install-recommends bzip2 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://busybox.net/downloads/busybox-$BUSYBOX_VERSION.tar.bz2" -o /tmp/busybox.tar.bz2 \
    && echo "$BUSYBOX_SHA256  /tmp/busybox.tar.bz2" | sha256sum -c - \
    && mkdir /busybox \
    && tar -xjf /tmp/busybox.tar.bz2 -C /busybox --strip-components=1
WORKDIR /busybox
COPY busybox.config.sh .
RUN make defconfig >/dev/null && sh busybox.config.sh
RUN make -j"$(nproc)" CC=rv32-cc HOSTCC=gcc AR="zig ar" SKIP_STRIP=y busybox \
    && make CC=rv32-cc HOSTCC=gcc AR="zig ar" SKIP_STRIP=y CONFIG_PREFIX=/rootfs install >/dev/null

FROM scratch AS busybox
COPY --from=busybox-build /rootfs/ /
