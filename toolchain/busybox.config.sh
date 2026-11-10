#!/bin/sh
# Adjust BusyBox's defconfig for shem guests: static, and without applets that
# do not build against current kernel headers or make no sense in a container.
set -e
set_option() { sed -i "s/^# $1 is not set$/$1=y/; s/^$1=.*/$1=y/" .config; }
unset_option() { sed -i "s/^$1=.*/# $1 is not set/" .config; }

set_option CONFIG_STATIC
unset_option CONFIG_TC  # needs CBQ, which Linux 6.8 headers removed
unset_option CONFIG_HWCLOCK  # needs settimeofday, which riscv32 lacks
