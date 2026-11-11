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
