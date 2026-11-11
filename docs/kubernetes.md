# Kubernetes

Kubernetes runs shem containers through a RuntimeClass, the way it runs WebAssembly with runwasi: the kubelet asks containerd for the `shem` runtime, and containerd starts the shim ([containerd.md](containerd.md)).

## On each node

Install the shim where containerd finds it.

```sh
sudo install containerd-shim-shem-v1 /usr/local/bin/
```

Add the runtime to containerd's configuration (`/etc/containerd/config.toml`, configuration version 3), and tell its image service that the runtime's images are for `linux/riscv32`.

```toml
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.shem]
  runtime_type = 'io.containerd.shem.v1'

[plugins.'io.containerd.cri.v1.images'.runtime_platforms.shem]
  platform = 'linux/riscv32'
```

Restart containerd.

```sh
sudo systemctl restart containerd
```

### Images for the node's platform

Released containerd (through 2.4) picks a pod's image for the node's platform, `linux/amd64` or `linux/arm64`, whatever its runtime class, so an image listed only for `linux/riscv32` is not found (`ErrImageNeverPull`, or a failed pull). List shem images under the node's platforms too; the guest is RV32 whichever entry containerd picks, and the shem runtime runs it:

- `shem push --platform linux/amd64,linux/arm64 IMAGE REMOTE` (and `shem save --platform ...`) list the same image under those platforms as well;
- `docker build --platform linux/riscv32,linux/amd64,linux/arm64` builds it for each, which for an image of RV32 programs gives the same files each time.

containerd's development branch picks images per runtime class instead (`runtime_platforms` above, with the kubelet's `RuntimeClassInImageCriApi` feature gate), which makes this unnecessary once released.

## In the cluster

Create the RuntimeClass.

```sh
kubectl apply -f examples/kubernetes/runtimeclass.yaml
```

Run a Pod on it.

```sh
kubectl apply -f examples/kubernetes/hello.yaml
```

Read its output.

```sh
kubectl logs shem-hello
```

Any shem image works, model images included, with the model server running on the node for them (see [containerd.md](containerd.md)). A test builds a kind cluster, configures its node this way, and runs an RV32 program in a Pod; CI runs it on every push.
