"""Prometheus metrics, in the text exposition format, for the API server and the registry.

A `Metrics` keeps counters, gauges and histograms by name and labels, behind a
lock, and renders them for a scraper's `GET /metrics`. It is as small as that
needs to be: no client library, and nothing it does not use.
"""

import threading
from collections import defaultdict

#: Seconds, for the latency of a request: an API call may take minutes
LATENCY = (0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def _labels(labels: dict) -> tuple:
    return tuple(sorted((key, str(value)) for key, value in labels.items()))


def _text(labels: tuple, extra: tuple = ()) -> str:
    pairs = [*labels, *extra]
    if not pairs:
        return ""
    quoted = ",".join(
        '{}="{}"'.format(k, v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")) for k, v in pairs
    )
    return "{" + quoted + "}"


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.help: dict[str, str] = {}
        self.kind: dict[str, str] = {}
        self.values: dict[str, dict[tuple, float]] = defaultdict(dict)
        #: Histograms: per labels, the bucket bounds, the counts at or below each, the sum
        self.histograms: dict[str, dict[tuple, list]] = defaultdict(dict)

    def describe(self, name: str, kind: str, text: str) -> None:
        self.kind[name], self.help[name] = kind, text

    def inc(self, name: str, amount: float = 1, /, **labels) -> None:
        with self.lock:
            values = self.values[name]
            key = _labels(labels)
            values[key] = values.get(key, 0) + amount

    def set(self, name: str, value: float, /, **labels) -> None:
        with self.lock:
            self.values[name][_labels(labels)] = value

    def observe(self, name: str, value: float, /, buckets: tuple = LATENCY, **labels) -> None:
        with self.lock:
            key = _labels(labels)
            entry = self.histograms[name].setdefault(key, [buckets, [0] * len(buckets), 0.0, 0])
            for at, bound in enumerate(entry[0]):
                if value <= bound:
                    entry[1][at] += 1
            entry[2] += value
            entry[3] += 1

    def render(self) -> str:
        """Everything, as Prometheus reads it."""
        lines: list[str] = []
        with self.lock:
            for name in sorted({*self.values, *self.histograms}):
                lines.append(f"# HELP {name} {self.help.get(name, name)}")
                kind = self.kind.get(name, "histogram" if name in self.histograms else "counter")
                lines.append(f"# TYPE {name} {kind}")
                for labels, value in sorted(self.values.get(name, {}).items()):
                    lines.append(f"{name}{_text(labels)} {value:g}")
                for labels, (buckets, counts, total, count) in sorted(self.histograms.get(name, {}).items()):
                    for bound, seen in zip(buckets, counts, strict=True):
                        lines.append(f"{name}_bucket{_text(labels, (('le', f'{bound:g}'),))} {seen}")
                    lines.append(f"{name}_bucket{_text(labels, (('le', '+Inf'),))} {count}")
                    lines.append(f"{name}_sum{_text(labels)} {total:g}")
                    lines.append(f"{name}_count{_text(labels)} {count}")
        return "\n".join(lines) + "\n"
