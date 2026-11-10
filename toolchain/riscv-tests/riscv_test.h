// riscv-tests environment for shem.
//
// Each test is a static Linux user-mode executable that reports through its
// exit status, so it runs as an ordinary container: `shem run rv32ui-add`.

#ifndef _ENV_SHEM_H
#define _ENV_SHEM_H

#define RVTEST_RV32U .macro init; .endm
#define RVTEST_RV64U .macro init; .endm

#define TESTNUM gp

#define RVTEST_CODE_BEGIN \
  .text;                  \
  .globl _start;          \
_start:                   \
  init;

#define RVTEST_CODE_END unimp

// Exit 0 on success. On failure exit with (TESTNUM << 1) | 1, the encoding
// upstream uses for tohost: the status is odd, so it never reads as success.
#define RVTEST_PASS \
  li a0, 0;         \
  li a7, 93;        \
  ecall

#define RVTEST_FAIL      \
  slli a0, TESTNUM, 1;   \
  ori a0, a0, 1;         \
  li a7, 93;             \
  ecall

#define EXTRA_DATA

#define RVTEST_DATA_BEGIN \
  EXTRA_DATA              \
  .data;                  \
  .align 4;               \
  .global begin_signature; \
begin_signature:

#define RVTEST_DATA_END \
  .align 4;             \
  .global end_signature; \
end_signature:

#endif
