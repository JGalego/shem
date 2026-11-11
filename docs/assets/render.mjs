// Regenerate the documentation visuals: node docs/assets/render.mjs
// Requires Node.js, Python with CairoSVG and Pillow, and Fontconfig with Noto Color Emoji.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const ink = "#29343a";
const muted = "#65716f";
const paper = "#fbf8f2";
const clay = "#a64f35";
const clayWash = "#f4e4d9";
const teal = "#216b63";
const tealWash = "#e3efea";
const line = "#bdc5bd";

const [emojiFamily, emojiFont] = execFileSync("fc-match", ["-f", "%{family}\n%{file}", "Noto Color Emoji"], { encoding: "utf8" }).trim().split("\n");
if (emojiFamily !== "Noto Color Emoji") throw new Error("Install Noto Color Emoji to render the diagram icons.");
const emojiImages = JSON.parse(execFileSync("python", ["-c", `
import base64, io, json, sys
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype(sys.argv[1], 109)
icons = {}
for emoji in json.load(sys.stdin):
    image = Image.new("RGBA", (160, 160))
    ImageDraw.Draw(image).text((0, 0), emoji, font=font, embedded_color=True)
    bounds = image.getbbox()
    if bounds is None:
        raise ValueError(f"No glyph rendered for {emoji}")
    image = image.crop(bounds)
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    icons[emoji] = base64.b64encode(output.getvalue()).decode("ascii")
json.dump(icons, sys.stdout)
`, emojiFont], { input: JSON.stringify(["🐳", "🦀", "🐍", "🤗", "📦", "🔥"]), encoding: "utf8" }));

function emoji(x, y, value, size = 24) {
  return `<image x="${x}" y="${y}" width="${size}" height="${size}" href="data:image/png;base64,${emojiImages[value]}" aria-hidden="true"/>`;
}

function escape(text) {
  return String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function text(x, y, value, size = 20, color = ink, anchor = "start", mono = false) {
  return `<text x="${x}" y="${y}" fill="${color}" font-family="${mono ? "DejaVu Sans Mono, monospace" : "DejaVu Sans, sans-serif"}" font-size="${size}" text-anchor="${anchor}">${escape(value)}</text>`;
}

function path(d, color = ink, width = 2, extra = "") {
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
}

function box(x, y, w, h, fill = paper, color = ink, dashed = false) {
  const d = `M ${x + 9} ${y + 1} Q ${x + w / 2} ${y - 2} ${x + w - 9} ${y}
    Q ${x + w + 1} ${y} ${x + w} ${y + 10} L ${x + w - 1} ${y + h - 9}
    Q ${x + w} ${y + h + 1} ${x + w - 10} ${y + h} Q ${x + w / 2} ${y + h - 2} ${x + 9} ${y + h + 1}
    Q ${x - 1} ${y + h} ${x} ${y + h - 10} L ${x + 1} ${y + 10} Q ${x} ${y} ${x + 9} ${y + 1} Z`;
  return `<path d="${d}" fill="${fill}" stroke="${color}" stroke-width="1.8" ${dashed ? 'stroke-dasharray="7 6"' : ""}/>` +
    (dashed ? "" : `\n    ${path(`M ${x + 12} ${y + 3} Q ${x + w / 2} ${y + 1} ${x + w - 10} ${y + 2} M ${x + 3} ${y + 12} L ${x + 2} ${y + h - 12}`, color, 0.6, 'opacity=".45"')}`);
}

function arrow(x1, y1, x2, y2, color = ink, dashed = false) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = [-0.48, 0.48].map((offset) => {
    return `M ${x2 - 9 * Math.cos(angle + offset)} ${y2 - 9 * Math.sin(angle + offset)} L ${x2} ${y2}`;
  }).join(" ");
  return path(`M ${x1} ${y1} Q ${(x1 + x2) / 2 + 1} ${(y1 + y2) / 2 - 2} ${x2} ${y2}`, color, 2, dashed ? 'stroke-dasharray="6 5"' : "") + path(head, color);
}

function dot(x1, y1, x2, y2, progress, color) {
  const x = x1 + (x2 - x1) * progress;
  const y = y1 + (y2 - y1) * progress;
  return `<circle cx="${x}" cy="${y}" r="5" fill="${color}" stroke="${paper}" stroke-width="2"/>`;
}

function network(x, y, active = false, progress = 0) {
  let svg = "";
  const rows = [0, 27, 54];
  for (let layer = 0; layer < 2; layer++) {
    for (const a of rows) {
      for (const b of rows) {
        svg += path(`M ${x + layer * 52} ${y + a} L ${x + (layer + 1) * 52} ${y + b}`, active ? clay : line, 1, 'opacity=".6"');
      }
    }
  }
  for (let layer = 0; layer < 3; layer++) {
    for (const row of rows) {
      const lit = active && layer === Math.min(2, Math.floor(progress * 3));
      svg += `<circle cx="${x + layer * 52}" cy="${y + row}" r="5.5" fill="${lit ? clay : paper}" stroke="${clay}" stroke-width="1.5"/>`;
    }
  }
  return svg;
}

function svg(height, title, description, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="${height}" viewBox="0 0 960 ${height}" role="img" aria-labelledby="title description">
  <title id="title">${escape(title)}</title>
  <desc id="description">${escape(description)}</desc>
  <rect width="960" height="${height}" fill="${paper}"/>
  ${body}
</svg>
`;
}

function architecture() {
  const body = [
    box(32, 69, 225, 70), emoji(53, 77, "🐳", 25), text(160, 98, "Docker CLI", 22, ink, "middle"),
    text(144, 124, "build / push / pull / run", 13, muted, "middle"),
    arrow(264, 89, 349, 89, muted), text(306, 78, "push", 12, muted, "middle"),
    arrow(349, 118, 264, 118, muted), text(306, 142, "pull", 12, muted, "middle"),
    arrow(144, 147, 144, 187), text(129, 175, "build", 13, muted, "end"),
    box(32, 194, 225, 70), text(144, 223, "OCI image", 22, ink, "middle"), text(144, 249, "files / optional state", 15, muted, "middle"),
    box(357, 69, 212, 70), text(463, 98, "OCI registry", 22, ink, "middle"),
    text(463, 124, "Docker Hub / GHCR", 14, muted, "middle"),
    path("M 240 140 L 240 164 Q 240 174 250 174 L 453 174 Q 463 174 463 184"),
    arrow(463, 184, 463, 194), text(359, 164, "run", 14, muted, "middle"),
    arrow(264, 229, 349, 229), text(306, 219, "rootfs", 13, muted, "middle"),
    box(357, 194, 212, 70), text(463, 223, "containerd shim", 20, ink, "middle"), text(463, 249, "io.containerd.shem.v1", 14, muted, "middle", true),
    box(694, 69, 233, 70), text(810, 98, "shem CLI", 22, ink, "middle"), text(810, 124, "run / boot / push / pull", 16, muted, "middle"),
    path("M 810 140 L 810 276 Q 810 286 800 286 L 554 286 Q 544 286 544 296"), arrow(544, 296, 544, 303),
    arrow(463, 264, 463, 303),
    box(32, 306, 537, 320, tealWash, teal),
    text(55, 341, "host = hypervisor", 27, teal),
    emoji(55, 350, "🦀", 21), text(84, 368, "Rust runtime", 16, muted),
    box(54, 395, 237, 114, paper, teal),
    text(72, 425, "Container mode", 20, teal),
    text(72, 456, "Linux syscalls", 19),
    text(72, 485, "files / processes", 17, muted),
    box(310, 395, 237, 114, paper, teal),
    text(328, 425, "VM mode", 20, teal),
    text(328, 456, "CSRs / traps", 19),
    text(328, 485, "timer / UART / syscon", 17, muted),
    text(55, 551, "scheduler / storage / snapshots", 21, teal),
    text(55, 592, "Vcpu interface", 20, ink, "start", true),
    box(694, 306, 233, 320, clayWash, clay),
    text(810, 341, "model = machine", 23, clay, "middle"),
    emoji(714, 351, "🐍", 22), text(745, 368, "Python / uv", 14, muted),
    emoji(839, 352, "🔥", 20), text(866, 368, "PyTorch", 14, muted),
    box(714, 392, 193, 109, paper, clay),
    text(810, 419, "CPU = weights", 20, clay, "middle"),
    network(758, 433),
    box(714, 518, 193, 86, paper, clay),
    text(810, 545, "RAM + registers", 20, clay, "middle"),
    text(810, 573, "recurrent state", 16, ink, "middle"),
    text(630, 366, "Unix socket", 15, muted, "middle"),
    arrow(576, 408, 686, 408, teal), text(630, 397, "run", 17, teal, "middle", true),
    arrow(686, 464, 576, 464, clay), text(630, 452, "exit", 17, clay, "middle", true),
    arrow(576, 544, 686, 544, muted), arrow(686, 563, 576, 563, muted),
    text(630, 527, "state I/O", 15, muted, "middle"),
    path("M 516 626 L 516 668 Q 516 678 526 678 L 683 678", muted, 1.8, 'stroke-dasharray="6 5"'),
    arrow(680, 678, 694, 678, muted),
    box(694, 649, 233, 65, paper, muted, true),
    text(810, 675, "or reference vCPU", 19, ink, "middle"),
    emoji(714, 684, "🦀", 19), text(747, 700, "RV32IM / Rust", 14, muted),
    text(32, 675, "guest: RV32 program or kernel", 18, ink),
    text(32, 705, "both vCPUs use the same snapshot format", 16, muted),
  ].join("\n  ");
  return svg(702, "shem architecture", "The Docker CLI builds OCI images and pushes or pulls them through a standard registry. Docker run reaches the shem containerd shim, which receives the image's root filesystem and drives the host. The shem CLI can drive the host directly. The Rust host provides Linux syscalls in container mode, or privileged instructions and devices in VM mode. A Unix socket connects it to the Python/PyTorch neural vCPU: CPU weights and register/memory state. A Rust reference interpreter is an alternative backend with the same snapshot format.", `<g transform="translate(0 -40)">${body}</g>`);
}

function dockerWorkflow() {
  const body = [
    box(32, 32, 330, 151),
    emoji(50, 47, "🐳", 26), text(88, 70, "Dockerfile", 22),
    text(50, 106, "FROM hf.co/Qwen/Qwen3-0.6B", 14, ink, "start", true),
    text(50, 137, 'CMD ["Hello"]', 15, muted, "start", true),
    text(50, 166, "shem's BuildKit frontend", 15, muted),
    arrow(370, 108, 468, 108), text(419, 91, "docker build", 13, ink, "middle", true),
    box(476, 32, 186, 238, clayWash, clay),
    emoji(493, 48, "📦", 24), text(581, 70, "OCI image", 20, clay, "middle"),
    text(569, 119, "model weights", 18, ink, "middle"),
    text(569, 148, "tokenizer", 16, muted, "middle"),
    text(569, 177, "optional session", 14, muted, "middle"),
    text(569, 235, "linux/riscv32", 14, muted, "middle", true),
    box(711, 32, 217, 65),
    emoji(728, 45, "🤗", 26), text(767, 58, "Hugging Face", 19),
    text(767, 81, "checkpoints", 14, muted),
    arrow(703, 65, 670, 65, clay), text(686, 51, "import", 12, clay, "middle"),
    arrow(670, 159, 776, 159, clay), text(723, 142, "docker push", 13, clay, "middle", true),
    arrow(776, 228, 670, 228, teal), text(723, 253, "docker pull", 13, teal, "middle", true),
    box(784, 113, 144, 157),
    text(856, 147, "OCI registry", 17, ink, "middle"),
    text(856, 191, "Docker Hub", 14, ink, "middle"),
    text(856, 230, "GHCR / ...", 15, muted, "middle"),
    path("M 569 278 L 569 292 Q 569 302 559 302 L 181 302 Q 171 302 171 312", teal),
    arrow(171, 312, 171, 322, teal),
    text(370, 294, "docker run --runtime io.containerd.shem.v1", 15, teal, "middle", true),
    box(32, 330, 277, 90, tealWash, teal),
    text(170, 366, "containerd + shem shim", 20, teal, "middle"),
    text(170, 397, "selects the image's driver", 16, muted, "middle"),
    arrow(317, 375, 414, 375, teal),
    box(422, 330, 506, 90, tealWash, teal),
    emoji(445, 362, "🐍", 26), text(484, 383, "Python / PyTorch session", 22),
  ].join("\n  ");
  return svg(452, "Model Dockerfile to registry to shem", "Hugging Face supplies model checkpoints to shem's BuildKit frontend. docker build produces a linux/riscv32 OCI image containing the model weights, tokenizer and optional session state. docker push and pull exchange it with an ordinary registry, such as Docker Hub or GHCR. docker run selects io.containerd.shem.v1 and starts a Python/PyTorch model session through the shem shim.", body);
}

function imageComparison() {
  const body = [
    box(32, 34, 430, 356, paper, teal),
    emoji(54, 45, "🐳", 25), text(93, 67, "Regular Docker image", 22, teal),
    box(54, 88, 386, 52),
    text(247, 120, "FROM python:3.12", 16, ink, "middle", true),
    arrow(247, 145, 247, 158, teal),
    box(54, 160, 386, 72, tealWash, teal),
    text(247, 186, "python:3.12", 19, teal, "middle", true),
    text(247, 214, "OS + Python runtime", 16, muted, "middle"),
    arrow(247, 237, 247, 249, teal),
    box(54, 251, 386, 59),
    text(247, 287, "app code + dependencies", 18, ink, "middle"),
    arrow(247, 315, 247, 324, teal),
    box(54, 326, 386, 48, tealWash, teal),
    emoji(69, 337, "📦", 25), text(261, 357, "OCI image", 21, teal, "middle"),

    box(498, 34, 430, 356, paper, clay),
    emoji(520, 45, "🤗", 25), text(559, 67, "LLM image", 22, clay),
    box(520, 88, 386, 52),
    text(713, 120, "FROM hf.co/Qwen/Qwen3.5-0.8B", 14, ink, "middle", true),
    arrow(713, 145, 713, 158, clay),
    box(520, 160, 386, 72, clayWash, clay),
    text(713, 186, "model weights + tokenizer", 18, clay, "middle"),
    text(713, 214, "imported from Hugging Face", 15, muted, "middle"),
    arrow(713, 237, 713, 249, clay),
    box(520, 251, 386, 59),
    text(713, 276, "ENTRYPOINT + prompt", 17, ink, "middle"),
    text(713, 299, "optional WARM session", 14, muted, "middle"),
    arrow(713, 315, 713, 324, clay),
    box(520, 326, 386, 48, clayWash, clay),
    emoji(535, 337, "📦", 25), text(727, 357, "OCI image", 21, clay, "middle"),
    text(480, 427, "Different bases; the same OCI image format", 19, muted, "middle"),
  ].join("\n  ");
  return svg(448, "Regular Docker images and LLM images", "A regular Docker image starts from a runtime such as python:3.12, then adds application code and dependencies. An LLM image starts from a Hugging Face checkpoint, which supplies model weights and a tokenizer, then adds its entrypoint, prompt and optional warmed session. Both are OCI images.", body);
}

const phases = [
  { label: "Load", seconds: 1.6, pc: "00", t0: 0, a0: 1, row: -1, route: "load" },
  { label: "Step", seconds: 1.6, pc: "04", t0: 6, a0: 1, row: 0, route: "cpu" },
  { label: "Step", seconds: 1.6, pc: "08", t0: 7, a0: 1, row: 1, route: "cpu" },
  { label: "Exit", seconds: 1.5, pc: "08", t0: 7, a0: 1, row: 2, route: "exit" },
  { label: "Host", seconds: 1.9, pc: "08", t0: 7, a0: 1, row: 2, route: "host" },
  { label: "Resume", seconds: 1.6, pc: "0c", t0: 7, a0: 3, row: 3, route: "resume" },
  { label: "Step", seconds: 1.6, pc: "10", t0: 8, a0: 3, row: 3, route: "cpu" },
];

function execution(phase, progress = 0, poster = false) {
  const activeCpu = phase.route === "cpu";
  const exited = ["exit", "host", "resume"].includes(phase.route);
  const body = [
    text(32, 71, "container mode", 15, muted),
    text(928, 71, poster ? "step / exit / resume" : phase.label.toLowerCase(), 18, poster ? muted : exited ? teal : clay, "end"),
    box(32, 99, 264, 289),
    text(52, 132, "RV32 program", 22),
    text(52, 160, "...", 18, muted, "start", true),
    ...["addi t0, zero, 6", "addi t0, t0, 1", "ecall", "addi t0, t0, 1"].map((instruction, row) => {
      const y = 191 + row * 33;
      return (phase.row === row ? `<rect x="44" y="${y - 23}" width="239" height="31" rx="4" fill="${exited ? tealWash : clayWash}"/>` : "") +
        text(52, y, instruction, 18, phase.row === row ? ink : muted, "start", true);
    }),
    text(52, 349, 'write args already set', 15, muted),
    box(338, 99, 289, 289, clayWash, clay),
    emoji(355, 109, "🐍", 24), text(495, 132, "model = machine", 23, clay, "middle"),
    box(356, 151, 252, 100, paper, clay),
    network(379, 174, activeCpu, progress),
    text(544, 192, "CPU", 21, clay, "middle"),
    text(544, 219, "weights", 16, muted, "middle"),
    box(356, 265, 252, 104, paper, clay),
    text(373, 292, "state", 17, clay),
    text(373, 320, `pc  0x${phase.pc}   t0  ${phase.t0}`, 18, ink, "start", true),
    text(373, 347, `a0  ${phase.a0}`, 18, ink, "start", true),
    text(590, 347, '"hi\\n"', 18, muted, "end", true),
    arrow(302, 209, 330, 209, phase.route === "load" ? clay : line),
    box(730, 99, 198, 289, tealWash, teal),
    text(829, 132, "host", 24, teal, "middle"),
    text(829, 185, "Linux syscall", 18, ink, "middle"),
    text(829, 215, "write", 20, teal, "middle", true),
    box(748, 252, 162, 85, paper, teal),
    text(762, 278, "stdout", 14, muted),
    text(762, 310, ["host", "resume"].includes(phase.route) || phase.a0 === 3 || poster ? "hi" : "", 24, teal, "start", true),
    arrow(635, 199, 722, 199, exited || poster ? clay : line),
    text(678, 183, "exit", 16, exited || poster ? clay : muted, "middle"),
    arrow(722, 300, 635, 300, phase.route === "resume" || poster ? teal : line),
    text(678, 328, "resume", 16, phase.route === "resume" || poster ? teal : muted, "middle"),
    text(482, 421, activeCpu || poster ? "1 forward pass = 1 instruction" : phase.route === "load" ? "image -> memory" : phase.route === "resume" ? "a0 = bytes written; pc += 4" : phase.route === "host" ? 'write(1, "hi\\n", 3)' : "ecall: no instruction retires", 18, exited ? teal : clay, "middle", true),
  ];
  if (phase.route === "load") body.push(dot(302, 209, 330, 209, progress, clay));
  if (phase.route === "exit") body.push(dot(635, 199, 722, 199, progress, clay));
  if (phase.route === "resume") body.push(dot(722, 300, 635, 300, progress, teal));
  return svg(411, "How shem runs a guest", "A program executes on the compiled RV32IM vCPU, one instruction per forward pass, updating register and memory state. ecall stops execution without retiring. The host writes hi to stdout, sets a0 to 3, advances the pc by four bytes, and resumes the guest.", `<g transform="translate(0 -40)">${body.join("\n  ")}</g>`);
}

writeFileSync(join(directory, "architecture.svg"), architecture());
writeFileSync(join(directory, "docker-workflow.svg"), dockerWorkflow());
writeFileSync(join(directory, "image-comparison.svg"), imageComparison());
writeFileSync(join(directory, "execution.svg"), execution(phases[5], 0, true));

const scratch = mkdtempSync(join(tmpdir(), "shem-visuals-"));
try {
  const frames = [];
  let previous;
  for (const phase of phases) {
    const count = Math.round(phase.seconds * 10);
    for (let frame = 0; frame < count; frame++) {
      const source = execution(phase, frame / (count - 1));
      if (source === previous) {
        frames.at(-1).milliseconds += 100;
        continue;
      }
      const filename = join(scratch, `${String(frames.length).padStart(3, "0")}.svg`);
      writeFileSync(filename, source);
      frames.push({ filename, milliseconds: 100 });
      previous = source;
    }
  }
  execFileSync("python", ["-c", `
import io, json, sys
import cairosvg
from PIL import Image

frames = []
durations = []
palette = None
for frame in json.load(sys.stdin):
    png = cairosvg.svg2png(url=frame["filename"])
    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGB")
    if palette is None:
        palette = image.quantize(colors=256)
    image = image.quantize(palette=palette, dither=Image.Dither.NONE)
    image.info.clear()
    frames.append(image)
    durations.append(frame["milliseconds"])
frames[0].save(sys.argv[1], save_all=True, append_images=frames[1:],
               duration=durations, loop=0, disposal=1, optimize=True)
`, join(directory, "execution.gif")], { input: JSON.stringify(frames), stdio: ["pipe", "inherit", "inherit"] });
  console.log("Rendered architecture.svg, docker-workflow.svg, image-comparison.svg, execution.svg and execution.gif");
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
