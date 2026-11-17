#!/bin/sh
exec zig cc -target riscv32-linux-musl -mcpu=generic_rv32+m+a+c -static "$@"
