"""The metrics' text format, and the registry's digests that outlast it."""

from shem_neural.metrics import Metrics
from shem_neural.registry import Digests


def test_counters_gauges_and_histograms_render_as_prometheus_reads_them():
    m = Metrics()
    m.describe("hits_total", "counter", "Hits")
    m.describe("depth", "gauge", "Depth")
    m.inc("hits_total", path="/a", status=200)
    m.inc("hits_total", 2, path="/a", status=200)
    m.set("depth", 3)
    m.observe("took_seconds", 0.3, buckets=(0.1, 1.0), path="/a")
    m.observe("took_seconds", 5, buckets=(0.1, 1.0), path="/a")
    text = m.render()
    assert '# HELP hits_total Hits\n# TYPE hits_total counter\nhits_total{path="/a",status="200"} 3\n' in text
    assert "# TYPE depth gauge\ndepth 3\n" in text
    assert 'took_seconds_bucket{path="/a",le="0.1"} 0' in text
    assert 'took_seconds_bucket{path="/a",le="1"} 1' in text
    assert 'took_seconds_bucket{path="/a",le="+Inf"} 2' in text
    assert 'took_seconds_sum{path="/a"} 5.3' in text and 'took_seconds_count{path="/a"} 2' in text


def test_label_values_are_escaped():
    m = Metrics()
    m.inc("odd_total", name='a "quoted"\nname\\')
    assert 'odd_total{name="a \\"quoted\\"\\nname\\\\"} 1' in m.render()


def test_digests_survive_a_restart_and_a_bad_file_is_forgotten(tmp_path):
    path = tmp_path / "cache" / "digests.json"
    key = (("model.safetensors", "ab" * 32), ("other.bin", "cd" * 32))
    first = Digests(path)
    first[key] = "ef" * 32
    assert Digests(path)[key] == "ef" * 32, "a new registry has what the old one learnt"
    assert len(Digests(path)) == 1
    path.write_text("not json")
    assert dict(Digests(path)) == {}
    path.write_text('[[[["f", "x"]], "not a digest"]]')
    assert dict(Digests(path)) == {}, "a value that is no digest is not trusted"
    assert dict(Digests(None)) == {}
