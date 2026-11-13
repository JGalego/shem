package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/client/llb/sourceresolver"
	"github.com/moby/buildkit/frontend/dockerfile/dockerfile2llb"
	digest "github.com/opencontainers/go-digest"
	ocispecs "github.com/opencontainers/image-spec/specs-go/v1"
)

const neural = "shem-neural:test"

func transpile(t *testing.T, src string, imported map[string]Imported) string {
	t.Helper()
	out, err := Transpile([]byte(src), imported, nil, neural)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestDockerStagesBuildOnTheBuildPlatform(t *testing.T) {
	out := transpile(t, `ARG V=1
FROM alpine:3.22 AS build
RUN echo hi > /x
FROM --platform=linux/arm64 alpine AS arm
FROM scratch
COPY --from=build /x /x
ENTRYPOINT ["/x"]
`, nil)
	want := `ARG V=1
FROM --platform=$BUILDPLATFORM alpine:3.22 AS build
RUN echo hi > /x
FROM --platform=linux/arm64 alpine AS arm
FROM scratch
COPY --from=build /x /x
ENTRYPOINT ["/x"]
`
	if out != want {
		t.Errorf("got\n%s\nwant\n%s", out, want)
	}
}

func TestModelsBecomeOneLayerPerBlock(t *testing.T) {
	src := "FROM hf.co/Qwen/Qwen3-0.6B:main AS qwen\nCMD [\"hi\"]\n"
	imports, err := Imports([]byte(src))
	if err != nil {
		t.Fatal(err)
	}
	if len(imports) != 1 || imports[0] != (Import{"shem-import-0", "Qwen/Qwen3-0.6B", "main"}) {
		t.Fatalf("imports %+v", imports)
	}
	imported := map[string]Imported{"shem-import-0": {Files: []string{
		"shem.json", "tokenizer.json", "embed.safetensors", "head.safetensors",
		"blocks/000-attention.safetensors", "blocks/001-mlp.safetensors",
	}, License: "Apache-2.0"}}
	out := transpile(t, src, imported)
	for _, line := range []string{
		"FROM scratch AS qwen",
		"COPY --link --from=shem-import-0 /out/model/shem.json /out/model/tokenizer.json /model/",
		"COPY --link --from=shem-import-0 /out/model/blocks/000-attention.safetensors /model/blocks/",
		"COPY --link --from=shem-import-0 /out/model/blocks/001-mlp.safetensors /model/blocks/",
		"COPY --link --from=shem-import-0 /out/model/embed.safetensors /model/",
		`LABEL dev.shem.machine="rv32im" dev.shem.model="/model"`,
		`LABEL org.opencontainers.image.source="https://huggingface.co/Qwen/Qwen3-0.6B"`,
		`LABEL org.opencontainers.image.licenses="Apache-2.0"`,
		`ENTRYPOINT ["shem:generate"]`,
		`CMD ["hi"]`,
	} {
		if !strings.Contains(out, line+"\n") {
			t.Errorf("missing %q in\n%s", line, out)
		}
	}
	if strings.Count(out, "COPY") != 5 {
		t.Errorf("want one COPY for the small files and one per weights file:\n%s", out)
	}
}

func TestModelDockerfileExamples(t *testing.T) {
	for _, example := range []struct {
		directory string
		model     string
		revision  string
	}{
		{"qwen3", "Qwen/Qwen3-0.6B", ""},
		{"qwen3.5", "Qwen/Qwen3.5-0.8B", ""},
		{"golem", "Qwen/Qwen3.5-0.8B", ""},
		{"smollm2", "HuggingFaceTB/SmolLM2-360M-Instruct", ""},
		{"granite-moe", "ibm-granite/granite-3.1-1b-a400m-instruct", ""},
		{"mamba", "state-spaces/mamba-130m-hf", ""},
		{"llama-gguf", "bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M"},
		{"gemma3", "google/gemma-3-1b-it", ""},
		{"handbook", "allenai/OLMo-2-0425-1B-Instruct", ""},
	} {
		t.Run(example.directory, func(t *testing.T) {
			directory := filepath.Join("..", "examples", example.directory)
			src, err := os.ReadFile(filepath.Join(directory, "Dockerfile"))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(string(src), "# syntax=ghcr.io/jgalego/shem/dockerfile:latest\n") {
				t.Fatal("model Dockerfile must select shem's BuildKit frontend")
			}
			imports, err := Imports(src)
			if err != nil {
				t.Fatal(err)
			}
			if len(imports) != 1 || imports[0] != (Import{"shem-import-0", example.model, example.revision}) {
				t.Fatalf("imports %+v", imports)
			}
			shemfile, err := os.ReadFile(filepath.Join(directory, "Shemfile"))
			if err != nil {
				t.Fatal(err)
			}
			imported := map[string]Imported{"shem-import-0": {Files: []string{
				"shem.json", "tokenizer.json", "embed.safetensors", "blocks/000-attention.safetensors",
			}}}
			got := transpile(t, string(src), imported)
			want := transpile(t, string(shemfile), imported)
			if got != want {
				t.Errorf("Dockerfile and Shemfile differ:\nDockerfile:\n%s\nShemfile:\n%s", got, want)
			}
			lintClean(t, got)
		})
	}
}

// lintClean runs BuildKit's checks on a transpiled Dockerfile, as docker
// build does: a warning would point at a line of a file the user never sees.
func lintClean(t *testing.T, dockerfile string) {
	t.Helper()
	var err error
	ctx := context.Background()
	source := llb.Scratch().File(llb.Mkfile("Dockerfile", 0o644, []byte(dockerfile)))
	sourceMap := llb.NewSourceMap(&source, "Dockerfile", "Dockerfile", []byte(dockerfile))
	if sourceMap.Definition, err = source.Marshal(ctx); err != nil {
		t.Fatal(err)
	}
	results, err := dockerfile2llb.DockerfileLint(ctx, []byte(dockerfile), dockerfile2llb.ConvertOpt{
		SourceMap:    sourceMap,
		MetaResolver: offline{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if results.Error != nil {
		t.Errorf("BuildKit can't read the Dockerfile: %s\n%s", results.Error.Message, dockerfile)
	}
	for _, warning := range results.Warnings {
		t.Errorf("BuildKit warns %s: %s\n%s", warning.RuleName, warning.Detail, dockerfile)
	}
}

// offline resolves every image to an empty one for the platform asked for,
// with no registry: in a real build the imports are named contexts.
type offline struct{}

func (offline) ResolveImageConfig(_ context.Context, ref string, opt sourceresolver.Opt) (string, digest.Digest, []byte, error) {
	platform := ocispecs.Platform{OS: "linux", Architecture: runtime.GOARCH}
	if opt.ImageOpt != nil && opt.ImageOpt.Platform != nil {
		platform = *opt.ImageOpt.Platform
	}
	config, err := json.Marshal(ocispecs.Image{Platform: platform})
	return ref, digest.FromBytes(config), config, err
}

func TestModelStagesTakeTheirOwnEntrypoint(t *testing.T) {
	for _, src := range []string{
		"FROM hf.co/HuggingFaceTB/SmolLM2-360M-Instruct\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"96\"]\n",
		"FROM hf.co/HuggingFaceTB/SmolLM2-360M-Instruct\nWARM --until=prompt\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"96\"]\n",
	} {
		out := transpile(t, src, map[string]Imported{"shem-import-0": {Files: []string{"shem.json"}}})
		if strings.Count(out, "ENTRYPOINT") != 1 || !strings.Contains(out, `ENTRYPOINT ["shem:generate", "--max-tokens", "96"]`) {
			t.Errorf("want the stage's own ENTRYPOINT alone in\n%s", out)
		}
		lintClean(t, out)
	}
	out := transpile(t, "FROM hf.co/HuggingFaceTB/SmolLM2-360M-Instruct\n", map[string]Imported{"shem-import-0": {Files: []string{"shem.json"}}})
	if !strings.Contains(out, `ENTRYPOINT ["shem:generate"]`) {
		t.Errorf("want the default ENTRYPOINT in\n%s", out)
	}
}

func TestManyBlocksShareLayers(t *testing.T) {
	var files []string
	for i := range 300 {
		files = append(files, fmt.Sprintf("blocks/%03d-mlp.safetensors", i))
	}
	out := transpile(t, "FROM hf.co/big/model\n", map[string]Imported{"shem-import-0": {Files: files}})
	if n := strings.Count(out, "COPY"); n > maxModelLayers {
		t.Errorf("%d layers, more than %d", n, maxModelLayers)
	}
}

func TestWarmPrimesASessionInTheNeuralImage(t *testing.T) {
	src := `FROM hf.co/Qwen/Qwen3.5-0.8B AS golem
ENTRYPOINT ["shem:generate", "--system", "You are a golem."]
WARM --until=prompt
CMD ["Who are you?"]
`
	out := transpile(t, src, map[string]Imported{"shem-import-0": {Files: []string{"shem.json"}}})
	for _, line := range []string{
		"FROM scratch AS shem-0-before-warm-0",
		"FROM --platform=$BUILDPLATFORM shem-neural:test AS shem-0-warm-0",
		"COPY --from=shem-0-before-warm-0 / /image",
		`RUN ["shem-neural-generate","--system","You are a golem.","--warm","/out/.shem","/image"]`,
		"FROM shem-0-before-warm-0 AS golem",
		"COPY --from=shem-0-warm-0 /out/.shem /.shem",
		`CMD ["Who are you?"]`,
	} {
		if !strings.Contains(out, line+"\n") {
			t.Errorf("missing %q in\n%s", line, out)
		}
	}
}

func TestCompiledLayersComeFromTheirContext(t *testing.T) {
	src := "FROM scratch\nCOPY --from=shem:rv32im / /model/compiled/rv32im/\nCOPY --from=shem:rv32im /config.json /x\n"
	names, err := Compiled([]byte(src))
	if err != nil || len(names) != 1 || names[0] != "rv32im" {
		t.Fatalf("compiled %v, %v", names, err)
	}
	out := transpile(t, src, nil)
	if strings.Count(out, "COPY --from=shem-compiled-rv32im ") != 2 || strings.Contains(out, "shem:rv32im") {
		t.Error(out)
	}
	if _, err := Compiled([]byte("FROM scratch\nCOPY --from=shem:x86 / /\n")); err == nil {
		t.Error("shem:x86 is no compiled layer")
	}
}

func TestKernelsBecomeLabels(t *testing.T) {
	out := transpile(t, "FROM scratch\nKERNEL /boot/Image console=ttyS0 quiet\n", nil)
	if !strings.Contains(out, `LABEL dev.shem.kernel="/boot/Image" dev.shem.bootargs="console=ttyS0 quiet"`) {
		t.Error(out)
	}
}

func TestShemStagesRefuseWhatTheyCannotDo(t *testing.T) {
	for _, src := range []string{
		"FROM scratch\nRUN echo no\n",
		"FROM scratch\nENTRYPOINT [\"/bin/sh\"]\nWARM --until=syscall:read\n",
		"FROM scratch\nENTRYPOINT [\"/bin/sh\"]\nWARM --until=prompt\n",
		"FROM alpine AS build\nWARM --until=prompt\nFROM scratch\n",
		"ARG ONLY=1\n",
	} {
		if _, err := Transpile([]byte(src), map[string]Imported{}, nil, neural); err == nil {
			t.Errorf("want an error for\n%s", src)
		}
	}
}

func TestHeredocsSurvive(t *testing.T) {
	out := transpile(t, "FROM alpine AS build\nRUN <<EOF\necho one\necho two\nEOF\nFROM scratch\n", nil)
	if !strings.Contains(out, "RUN <<EOF\necho one\necho two\nEOF\n") {
		t.Error(out)
	}
}

func TestModelPackArtifactsLandAtTheRoot(t *testing.T) {
	src := "FROM ghcr.io/team/model:modelpack AS weights\nFROM weights\nCMD [\"hi\"]\n"
	imported := map[string]Imported{"shem-artifact-0": {
		Files: []string{"config.json", "model.safetensors", "shem.json", "tokenizer.json"},
		Model: "/",
	}}
	artifacts := map[string]string{"ghcr.io/team/model:modelpack": "shem-artifact-0"}
	out, err := Transpile([]byte(src), imported, artifacts, neural)
	if err != nil {
		t.Fatal(err)
	}
	want := `FROM scratch AS weights
COPY --link --from=shem-artifact-0 /out/artifact/config.json /out/artifact/shem.json /out/artifact/tokenizer.json /
COPY --link --from=shem-artifact-0 /out/artifact/model.safetensors /
LABEL dev.shem.machine="rv32im" dev.shem.model="/"
LABEL org.opencontainers.image.base.name="ghcr.io/team/model:modelpack"
ENTRYPOINT ["shem:generate"]
FROM weights
CMD ["hi"]
`
	if out != want {
		t.Errorf("got\n%s\nwant\n%s", out, want)
	}
}
