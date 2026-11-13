# Kubernetes

Kubernetes runs shem containers through a RuntimeClass, the way it runs WebAssembly with runwasi: the kubelet asks containerd for the `shem` runtime, and containerd starts the shim ([containerd.md](containerd.md)).

## On each node

Install the shim on containerd's `PATH`:

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

containerd through 2.4 selects images by the node's platform, even for a different runtime class. An image listed only for `linux/riscv32` can fail to pull. Also list it for the node's platform:

```sh
shem push --platform linux/amd64,linux/arm64 hello ghcr.io/your-user/hello:latest
```

`shem save --platform ...` works for local imports. With Docker, use `docker build --platform linux/riscv32,linux/amd64,linux/arm64`; the guest remains RV32.

containerd's development branch supports selection by runtime class using `runtime_platforms` above and the kubelet's `RuntimeClassInImageCriApi` feature gate.

## In the cluster

From the repository checkout:

```sh
kubectl apply -f examples/kubernetes/runtimeclass.yaml
kubectl apply -f examples/kubernetes/hello.yaml
kubectl logs shem-hello
```

Model Pods also require a [model server](models.md#sessions-under-docker) on the node. `just kubernetes-test` configures a kind node and runs the hello Pod.

## Model images as image volumes

A model image's `/model` is a Hugging Face checkpoint. Mount it as an [image volume](https://kubernetes.io/docs/concepts/storage/volumes/#image) to use another inference server without running shem.

Publish for the node's platform and enable the `ImageVolume` feature gate where required:

```sh
shem build -t smollm2 examples/smollm2
shem push --platform linux/amd64 smollm2 ghcr.io/your-user/smollm2:shem
```

Pod spec fragment:

```yaml
volumes:
- name: model
  image:
    reference: ghcr.io/your-user/smollm2:shem
containers:
- name: server
  # Set the server image and point it at /models/model.
  volumeMounts:
  - name: model
    mountPath: /models
    readOnly: true
```

The [complete Pod example](../examples/kubernetes/model-volume.yaml) loads the checkpoint with transformers. Replace `your-user` in the manifest before applying it. `just kubernetes-model-volume-test` checks volume mounting in kind.
