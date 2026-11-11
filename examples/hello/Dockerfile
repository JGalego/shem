# The hello program as an ordinary Docker image for RV32, to run on shem:
#   docker build --platform linux/riscv32 -t hello examples/hello
#   docker run --runtime io.containerd.shem.v1 --platform linux/riscv32 hello
# The build stage runs on the host's platform and cross-compiles with Zig.
FROM --platform=$BUILDPLATFORM alpine:3.22 AS build
RUN apk add --no-cache zig
WORKDIR /src
COPY hello.c .
RUN zig cc -target riscv32-linux-musl -mcpu=generic_rv32+m+a -static -O2 -o hello hello.c

FROM scratch
COPY --from=build /src/hello /bin/hello
ENTRYPOINT ["/bin/hello"]
CMD ["from", "a", "shem", "image"]
