"""A neural vCPU for shem: an RV32IM CPU compiled into network weights.

`isa` describes the instruction set as data, `compile` turns that description
into the network's weight tensors, `cpu` is the network, one instruction per
forward pass, `vcpu` wraps it in the interface the shem host drives, and
`server` exposes vCPUs to the host over a Unix socket.
"""
