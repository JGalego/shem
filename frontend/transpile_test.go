package main

import (
	"strings"
	"testing"
)

const neural = "shem-neural:test"

func transpile(t *testing.T, src string, imported map[string]Imported) string {
	t.Helper()
	out, err := Transpile([]byte(src), imported, neural)
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

func TestManyBlocksShareLayers(t *testing.T) {
	var files []string
	for i := range 300 {
		files = append(files, "blocks/"+strings.Repeat("0", 3-len(itoa(i)))+itoa(i)+"-mlp.safetensors")
	}
	out := transpile(t, "FROM hf.co/big/model\n", map[string]Imported{"shem-import-0": {Files: files}})
	if n := strings.Count(out, "COPY"); n > maxModelLayers {
		t.Errorf("%d layers, more than %d", n, maxModelLayers)
	}
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
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
		"COPY --from=shem-0-before-warm-0 /model /model",
		`RUN ["shem-neural-generate","--system","You are a golem.","--warm","/out/.shem","/model"]`,
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
		if _, err := Transpile([]byte(src), map[string]Imported{}, neural); err == nil {
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
