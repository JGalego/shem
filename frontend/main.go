// shem's BuildKit frontend: `docker build` for Shemfiles.
//
//	# syntax=ghcr.io/jgalego/shem/dockerfile:latest
//	FROM hf.co/Qwen/Qwen3.5-0.8B
//	ENTRYPOINT ["shem:generate", "--system", "You are a golem."]
//	WARM --until=prompt
//
// Imports become named contexts; transpile.go lowers shem instructions to a
// Dockerfile built by BuildKit's standard frontend. The default platform is
// linux/riscv32.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/containerd/platforms"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/frontend/gateway/grpcclient"
	"github.com/moby/buildkit/solver/pb"
	"github.com/moby/buildkit/util/appcontext"
	ocispecs "github.com/opencontainers/image-spec/specs-go/v1"
)

// version is the frontend's release, which picks the shem-neural image it
// runs; release builds set it.
var version = "latest"

func main() {
	if err := grpcclient.RunFromEnvironment(appcontext.Context(), build); err != nil {
		fmt.Fprintf(os.Stderr, "shem frontend: %v\n", err)
		os.Exit(1)
	}
}

func build(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	filename := opts["filename"]
	if filename == "" {
		filename = "Dockerfile"
	}
	src, err := readShemfile(ctx, c, filename)
	if err != nil {
		return nil, err
	}
	neuralImage := opts["build-arg:SHEM_NEURAL_IMAGE"]
	if neuralImage == "" {
		neuralImage = "ghcr.io/jgalego/shem-neural:" + version
	}
	buildPlatform := platforms.DefaultSpec()
	if workers := c.BuildOpts().Workers; len(workers) > 0 && len(workers[0].Platforms) > 0 {
		buildPlatform = workers[0].Platforms[0]
	}

	// The options BuildKit gave this frontend, minus the ones that would make
	// the Dockerfile frontend hand the build back to it
	forward := map[string]string{}
	for key, value := range opts {
		if key != "source" && key != "cmdline" {
			forward[key] = value
		}
	}
	inputs := map[string]*pb.Definition{}
	imported := map[string]Imported{}
	imports, err := Imports(src)
	if err != nil {
		return nil, err
	}
	for _, imp := range imports {
		def, model, err := importModel(ctx, c, imp, neuralImage, buildPlatform)
		if err != nil {
			return nil, fmt.Errorf("importing %s: %w", imp.Repo, err)
		}
		imported[imp.Name] = model
		inputs[imp.Name] = def
		forward["context:"+imp.Name] = "input:" + imp.Name
	}

	compiled, err := Compiled(src)
	if err != nil {
		return nil, err
	}
	for _, name := range compiled {
		def, err := compileLayer(ctx, c, name, neuralImage, buildPlatform)
		if err != nil {
			return nil, fmt.Errorf("compiling shem:%s: %w", name, err)
		}
		inputs[CompiledContext(name)] = def
		forward["context:"+CompiledContext(name)] = "input:" + CompiledContext(name)
	}

	dockerfile, err := Transpile(src, imported, neuralImage)
	if err != nil {
		return nil, err
	}
	def, err := llb.Scratch().File(llb.Mkfile("Dockerfile", 0o644, []byte(dockerfile))).Marshal(ctx)
	if err != nil {
		return nil, err
	}
	inputs["dockerfile"] = def.ToPB()
	forward["filename"] = "Dockerfile"
	if forward["platform"] == "" {
		forward["platform"] = "linux/riscv32"
	}
	return c.Solve(ctx, client.SolveRequest{
		Frontend:       "dockerfile.v0",
		FrontendOpt:    forward,
		FrontendInputs: inputs,
	})
}

func readShemfile(ctx context.Context, c client.Client, filename string) ([]byte, error) {
	local := llb.Local("dockerfile",
		llb.FollowPaths([]string{filename}),
		llb.SessionID(c.BuildOpts().SessionID),
		llb.SharedKeyHint("shemfile"),
		llb.WithCustomName("[shem] loading "+filename),
	)
	def, err := local.Marshal(ctx)
	if err != nil {
		return nil, err
	}
	res, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, err
	}
	ref, err := res.SingleRef()
	if err != nil {
		return nil, err
	}
	return ref.ReadFile(ctx, client.ReadRequest{Filename: filename})
}

// compileLayer builds `name` in shem-neural on the build platform and returns
// its weights, config and model card as a named context.
func compileLayer(ctx context.Context, c client.Client, name, neuralImage string, platform ocispecs.Platform) (*pb.Definition, error) {
	out := path.Join("/out/compiled", name)
	compiled := llb.Image(neuralImage, llb.Platform(platform), llb.WithMetaResolver(c)).Run(
		llb.Args([]string{"shem-neural-export", out, "--version", version}),
		llb.WithCustomName("[shem] compiling shem:"+name),
	).Root()
	files := llb.Scratch().File(
		llb.Copy(compiled, out+"/", "/", &llb.CopyInfo{CopyDirContentsOnly: true}),
		llb.WithCustomName("[shem] the compiled layer shem:"+name),
	)
	def, err := files.Marshal(ctx, llb.Platform(platform))
	if err != nil {
		return nil, err
	}
	return def.ToPB(), nil
}

// importModel runs in shem-neural on the build platform and returns the model
// files and declared licence. Downloads are cached; gated models use the
// optional HF_TOKEN build secret.
func importModel(ctx context.Context, c client.Client, imp Import, neuralImage string, platform ocispecs.Platform) (*pb.Definition, Imported, error) {
	var model Imported
	args := []string{"shem-neural-import", imp.Repo, importDir}
	if imp.Revision != "" {
		args = append(args, "--revision", imp.Revision)
	}
	// Resolving the image's config gives the run its environment, PATH included
	state := llb.Image(neuralImage, llb.Platform(platform), llb.WithMetaResolver(c)).Run(
		llb.Args(args),
		llb.AddMount("/root/.cache/huggingface", llb.Scratch(), llb.AsPersistentCacheDir("shem-huggingface", llb.CacheMountShared)),
		llb.AddSecret("/run/secrets/HF_TOKEN", llb.SecretID("HF_TOKEN"), llb.SecretAsEnv(true), llb.SecretOptional),
		llb.WithCustomName("[shem] importing "+imp.Repo),
	).Root()
	def, err := state.Marshal(ctx, llb.Platform(platform))
	if err != nil {
		return nil, model, err
	}
	res, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB(), Evaluate: true})
	if err != nil {
		return nil, model, err
	}
	ref, err := res.SingleRef()
	if err != nil {
		return nil, model, err
	}
	for _, dir := range []string{"", "blocks"} {
		entries, err := ref.ReadDir(ctx, client.ReadDirRequest{Path: path.Join(importDir, dir)})
		if err != nil {
			return nil, model, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				model.Files = append(model.Files, strings.TrimPrefix(path.Join(dir, entry.GetPath()), "/"))
			}
		}
	}
	manifest, err := ref.ReadFile(ctx, client.ReadRequest{Filename: path.Join(importDir, "shem.json")})
	if err != nil {
		return nil, model, err
	}
	var parsed struct {
		Source struct {
			SPDX string `json:"spdx"`
		} `json:"source"`
	}
	if err := json.Unmarshal(manifest, &parsed); err != nil {
		return nil, model, fmt.Errorf("the model's manifest: %w", err)
	}
	model.License = parsed.Source.SPDX
	return def.ToPB(), model, nil
}
