package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/moby/buildkit/frontend/dockerfile/parser"
)

// Lower Shemfiles to Dockerfiles: model imports, ModelPack artifacts and
// compiled layers use named contexts, WARM adds a priming stage, and KERNEL
// becomes config labels. Ordinary Docker stages run on the build platform.

const (
	hfPrefix  = "hf.co/"
	modelDir  = "/model"
	importDir = "/out/model"
	// artifactDir is where a ModelPack artifact's files are laid out
	artifactDir = "/out/artifact"
	// imageDir is where WARM sees the image's filesystem: the driver finds
	// the model in it, at /model or, in a ModelPack artifact's twin, at /
	imageDir       = "/image"
	maxModelLayers = 96
)

// An Import is a model a Shemfile builds on, which the frontend imports into
// the named context Name.
type Import struct {
	Name, Repo, Revision string
}

// An Imported model: its files, relative to the model's directory, the SPDX
// licence its model card declares, if any, and for a ModelPack artifact,
// where its model is once its files are at the image's root.
type Imported struct {
	Files   []string
	License string
	Model   string
}

type stage struct {
	from  *parser.Node
	nodes []*parser.Node
	base  string
	name  string
}

func parse(src []byte) (globals []*parser.Node, stages []stage, err error) {
	result, err := parser.Parse(bytes.NewReader(src))
	if err != nil {
		return nil, nil, err
	}
	for _, node := range result.AST.Children {
		if strings.EqualFold(node.Value, "from") {
			st := stage{from: node}
			if node.Next != nil {
				st.base = node.Next.Value
				if as := node.Next.Next; as != nil && strings.EqualFold(as.Value, "as") && as.Next != nil {
					st.name = as.Next.Value
				}
			}
			stages = append(stages, st)
		} else if len(stages) == 0 {
			globals = append(globals, node)
		} else {
			stages[len(stages)-1].nodes = append(stages[len(stages)-1].nodes, node)
		}
	}
	if len(stages) == 0 {
		return nil, nil, fmt.Errorf("the Shemfile has no FROM")
	}
	return globals, stages, nil
}

func hfReference(base string) (repo, revision string, ok bool) {
	rest, ok := strings.CutPrefix(base, hfPrefix)
	if !ok {
		return "", "", false
	}
	repo, revision, _ = strings.Cut(rest, ":")
	return repo, revision, true
}

// compiledPrefix names the compiled layers shem builds, as in COPY --from=shem:rv32im.
const compiledPrefix = "shem:"

// CompiledContext is the named context that holds the compiled layer `name`.
func CompiledContext(name string) string {
	return "shem-compiled-" + name
}

// Compiled lists the compiled layers a Shemfile copies, each once.
func Compiled(src []byte) ([]string, error) {
	_, stages, err := parse(src)
	if err != nil {
		return nil, err
	}
	var names []string
	seen := map[string]bool{}
	for _, st := range stages {
		for _, node := range st.nodes {
			if !strings.EqualFold(node.Value, "copy") {
				continue
			}
			if name, ok := strings.CutPrefix(flag(node, "from"), compiledPrefix); ok && !seen[name] {
				if name != "rv32im" {
					return nil, fmt.Errorf("line %d: shem:%s is no compiled layer; there is shem:rv32im", node.StartLine, name)
				}
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names, nil
}

// Imports lists the models a Shemfile builds on, in order.
func Imports(src []byte) ([]Import, error) {
	_, stages, err := parse(src)
	if err != nil {
		return nil, err
	}
	var imports []Import
	for _, st := range stages {
		if repo, revision, ok := hfReference(st.base); ok {
			imports = append(imports, Import{Name: fmt.Sprintf("shem-import-%d", len(imports)), Repo: repo, Revision: revision})
		}
	}
	return imports, nil
}

// Transpile writes the Dockerfile for a Shemfile, given what each import
// holds (by import name), the context name of each stage base that is a
// ModelPack artifact, and the image that runs shem's neural side.
func Transpile(src []byte, imported map[string]Imported, artifacts map[string]string, neuralImage string) (string, error) {
	globals, stages, err := parse(src)
	if err != nil {
		return "", err
	}
	var out strings.Builder
	write := func(format string, args ...any) { fmt.Fprintf(&out, format+"\n", args...) }
	for _, node := range globals {
		write("%s", original(node))
	}
	imports := 0
	for i, st := range stages {
		last := i == len(stages)-1
		repo, _, isModel := hfReference(st.base)
		artifact, isArtifact := artifacts[st.base]
		shem := last || isModel || isArtifact || st.base == "scratch"
		warms := 0
		for _, node := range st.nodes {
			if strings.EqualFold(node.Value, "warm") {
				warms++
			}
		}
		// A stage with WARM is split at each one: the parts before have names
		// of their own, and the last part has the stage's name
		part := func(k int) string {
			if k == warms {
				return st.name
			}
			return fmt.Sprintf("shem-%d-before-warm-%d", i, k)
		}
		as := func(name string) string {
			if name == "" {
				return ""
			}
			return " AS " + name
		}
		flags := strings.Join(st.from.Flags, " ")
		if flags != "" {
			flags += " "
		}

		var entrypoint []string
		switch {
		case isModel:
			name := fmt.Sprintf("shem-import-%d", imports)
			imports++
			model, ok := imported[name]
			if !ok {
				return "", fmt.Errorf("no files for the import of %s", repo)
			}
			write("FROM scratch%s", as(part(0)))
			for _, line := range modelCopies(name, model.Files, importDir, modelDir) {
				write("%s", line)
			}
			write(`LABEL dev.shem.machine="rv32im" dev.shem.model="%s"`, modelDir)
			write(`LABEL org.opencontainers.image.source=%s`, jsonString("https://huggingface.co/"+repo))
			if model.License != "" {
				write(`LABEL org.opencontainers.image.licenses=%s`, jsonString(model.License))
			}
			write(`ENTRYPOINT ["shem:generate"]`)
			entrypoint = []string{"shem:generate"}
		case isArtifact:
			// The artifact's files at the root, as in the twin shem makes of it
			model, ok := imported[artifact]
			if !ok {
				return "", fmt.Errorf("no files for the ModelPack artifact %s", st.base)
			}
			write("FROM scratch%s", as(part(0)))
			for _, line := range modelCopies(artifact, model.Files, artifactDir, "/") {
				write("%s", line)
			}
			write(`LABEL dev.shem.machine="rv32im" dev.shem.model=%s`, jsonString(model.Model))
			write(`LABEL org.opencontainers.image.base.name=%s`, jsonString(st.base))
			write(`ENTRYPOINT ["shem:generate"]`)
			entrypoint = []string{"shem:generate"}
		case !shem && !strings.Contains(flags, "--platform"):
			write("FROM --platform=$BUILDPLATFORM %s%s%s", flags, st.base, as(st.name))
		default:
			write("FROM %s%s%s", flags, st.base, as(part(0)))
		}

		warm := 0
		for _, node := range st.nodes {
			switch strings.ToLower(node.Value) {
			case "entrypoint":
				entrypoint = nil
				for arg := node.Next; arg != nil; arg = arg.Next {
					entrypoint = append(entrypoint, arg.Value)
				}
				write("%s", original(node))
			case "run":
				if shem {
					return "", fmt.Errorf("line %d: RUN is not allowed in a shem stage: build in a Docker stage and COPY --from it", node.StartLine)
				}
				write("%s", original(node))
			case "warm":
				until := flag(node, "until")
				if !shem {
					return "", fmt.Errorf("line %d: WARM is for shem stages", node.StartLine)
				}
				if until != "prompt" {
					return "", fmt.Errorf("line %d: docker build can WARM model images (--until=prompt); build programs and VMs that WARM with shem build", node.StartLine)
				}
				if len(entrypoint) == 0 || entrypoint[0] != "shem:generate" {
					return "", fmt.Errorf("line %d: WARM --until=prompt needs ENTRYPOINT [\"shem:generate\", ...]", node.StartLine)
				}
				warmName := fmt.Sprintf("shem-%d-warm-%d", i, warm)
				command := append([]string{"shem-neural-generate"}, entrypoint[1:]...)
				command = append(command, "--warm", "/out/.shem", imageDir)
				write("FROM --platform=$BUILDPLATFORM %s AS %s", neuralImage, warmName)
				write("COPY --from=%s / %s", part(warm), imageDir)
				write("RUN %s", jsonArray(command))
				warm++
				write("FROM %s%s", part(warm-1), as(part(warm)))
				write("COPY --from=%s /out/.shem /.shem", warmName)
			case "kernel":
				words := strings.Fields(strings.TrimSpace(original(node)[len("KERNEL"):]))
				if len(words) == 0 {
					return "", fmt.Errorf("line %d: KERNEL needs a path", node.StartLine)
				}
				label := fmt.Sprintf("LABEL dev.shem.kernel=%s", jsonString(words[0]))
				if len(words) > 1 {
					label += " dev.shem.bootargs=" + jsonString(strings.Join(words[1:], " "))
				}
				write("%s", label)
			case "copy":
				text := original(node)
				if name, ok := strings.CutPrefix(flag(node, "from"), compiledPrefix); ok {
					text = strings.Replace(text, "--from="+compiledPrefix+name, "--from="+CompiledContext(name), 1)
				}
				write("%s", text)
			default:
				write("%s", original(node))
			}
		}
	}
	return out.String(), nil
}

// modelCopies copies an import's files, from `from` in its context to `to`:
// the small files together, then each weights file in a layer of its own, or
// a few together when there would otherwise be too many layers, as shem
// build does.
func modelCopies(name string, files []string, from, to string) []string {
	var small, weights []string
	for _, file := range files {
		if strings.HasSuffix(file, ".safetensors") {
			weights = append(weights, file)
		} else {
			small = append(small, file)
		}
	}
	sort.Strings(small)
	sort.Strings(weights)
	var lines []string
	copyFiles := func(group []string, dest string) {
		sources := make([]string, len(group))
		for i, file := range group {
			sources[i] = path.Join(from, file)
		}
		lines = append(lines, fmt.Sprintf("COPY --link --from=%s %s %s/", name, strings.Join(sources, " "), strings.TrimSuffix(dest, "/")))
	}
	if len(small) > 0 {
		copyFiles(small, to)
	}
	perLayer := max(1, (len(weights)+maxModelLayers-2)/(maxModelLayers-1))
	for start := 0; start < len(weights); start += perLayer {
		group := weights[start:min(start+perLayer, len(weights))]
		// A COPY has one destination, so a group stays within one directory
		for len(group) > 0 {
			dir := path.Dir(group[0])
			n := 1
			for n < len(group) && path.Dir(group[n]) == dir {
				n++
			}
			copyFiles(group[:n], path.Join(to, dir))
			group = group[n:]
		}
	}
	return lines
}

// original is a node's source text, with any heredocs it carries.
func original(node *parser.Node) string {
	text := node.Original
	for _, heredoc := range node.Heredocs {
		text += "\n" + strings.TrimSuffix(heredoc.Content, "\n") + "\n" + heredoc.Name
	}
	return text
}

func flag(node *parser.Node, name string) string {
	for _, f := range node.Flags {
		if value, ok := strings.CutPrefix(f, "--"+name+"="); ok {
			return value
		}
	}
	return ""
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func jsonArray(items []string) string {
	b, _ := json.Marshal(items)
	return string(b)
}
