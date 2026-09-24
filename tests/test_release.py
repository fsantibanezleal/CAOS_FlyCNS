"""The MaleCNS rules, exercised on the synthetic release (see conftest.py for what each body is there to test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flycns.compiled import read_compiled
from flycns.release import ReleaseError, check_table, compile_malecns_v1
from flycns.release.malecns_v1 import SIGN_BY_TRANSMITTER, retained, sign_of


def compile_synthetic(release, out):
    directory, tables = release
    compile_malecns_v1(directory, out, tables=tables)
    return read_compiled(out)


def index_of(compiled, body):
    return int(np.flatnonzero(compiled["neuron_body_id"] == body)[0])


def test_rejects_a_table_whose_hash_differs(synthetic_release):
    directory, tables = synthetic_release
    path = directory / tables["weights"].file_name
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ReleaseError) as caught:
        check_table(directory, tables["weights"])
    message = str(caught.value)
    assert tables["weights"].file_name in message
    assert tables["weights"].sha256 in message
    assert "differs" in message


def test_keeps_superclass_bodies_once(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    bodies = compiled["neuron_body_id"].tolist()
    assert bodies == sorted(bodies)
    assert 900 not in bodies
    assert len(bodies) == len(set(bodies)) == 13
    duplicated = pd.DataFrame({"bodyId": [1, 1], "superclass": ["cb_intrinsic", "cb_intrinsic"]})
    with pytest.raises(ValueError):
        retained(duplicated)


def test_keeps_edges_between_retained_neurons_only(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    indptr, indices, counts = compiled["csr_indptr"], compiled["csr_indices"], compiled["csr_count"]
    assert compiled.n_edges == 8
    pairs = {}
    for pre in range(compiled.n_neurons):
        for k in range(indptr[pre], indptr[pre + 1]):
            pairs[(int(compiled["neuron_body_id"][pre]), int(compiled["neuron_body_id"][indices[k]]))] = int(counts[k])
    assert pairs[(300, 301)] == 15            # two rows of the same pair, summed
    assert pairs[(301, 400)] == 65535         # 70,000 synapses, saturated and counted
    assert compiled.counts["saturated_counts"] == 1
    assert all(900 not in pair for pair in pairs)
    for pre in range(compiled.n_neurons):     # targets ascend within a row
        row = indices[indptr[pre]:indptr[pre + 1]]
        assert np.all(np.diff(row) > 0)


def test_position_falls_back_in_order(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    source = compiled["neuron_position_source"]
    position = compiled["neuron_position_um"]
    names = compiled.strings["position_source"]
    assert names[source[index_of(compiled, 300)]] == "soma"
    assert names[source[index_of(compiled, 301)]] == "to_soma"
    assert names[source[index_of(compiled, 200)]] == "synapse_centroid"
    assert names[source[index_of(compiled, 304)]] == "none"
    np.testing.assert_allclose(position[index_of(compiled, 300)], [560.0, 240.0, 240.0])
    np.testing.assert_allclose(position[index_of(compiled, 301)], [160.0, 248.0, 320.0])
    np.testing.assert_allclose(position[index_of(compiled, 200)], [168.0, 248.0, 88.0], rtol=1e-6)
    assert np.all(np.isnan(position[index_of(compiled, 304)]))


@pytest.mark.parametrize("transmitter, sign", [
    ("acetylcholine", 1), ("gaba", -1), ("glutamate", -1), ("histamine", -1),
    ("dopamine", 1), ("octopamine", 1), ("serotonin", 1), ("unclear", 0), ("", 0),
])
def test_sign_rule(transmitter, sign):
    assert sign_of(transmitter) == sign
    assert set(SIGN_BY_TRANSMITTER) == {"acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine",
                                        "serotonin"}


def test_sign_rule_applied_in_the_compiled_graph(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    sign = compiled["neuron_sign"]
    nt = compiled.strings["transmitter"]
    source = compiled.strings["nt_source"]
    assert sign[index_of(compiled, 200)] == -1 and nt[compiled["neuron_nt"][index_of(compiled, 200)]] == "histamine"
    assert sign[index_of(compiled, 302)] == 1
    assert sign[index_of(compiled, 303)] == 0 and source[compiled["neuron_nt_source"][index_of(compiled, 303)]] == "none"
    assert sign[index_of(compiled, 301)] == 1
    assert source[compiled["neuron_nt_source"][index_of(compiled, 301)]] == "predicted"


def test_sides_and_partitions(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    side = compiled.strings["side"]
    partition = compiled.strings["partition"]
    expect = {100: ("right", "optic_lobe_right"), 103: ("left", "optic_lobe_left"),
              200: ("right", "optic_lobe_right"), 300: ("left", "central_brain"),
              301: ("right", "central_brain"), 400: ("midline", "nerve_cord")}
    for body, (s, p) in expect.items():
        i = index_of(compiled, body)
        assert side[compiled["neuron_side"][i]] == s
        assert partition[compiled["neuron_partition"][i]] == p


def test_photoreceptor_joins_the_column_it_drives_most(synthetic_release, tmp_path):
    compiled = compile_synthetic(synthetic_release, tmp_path / "out")
    column = compiled["neuron_column"]
    hexes = compiled["column_hex"]
    column_side = compiled["column_side"]
    for body, expected in ((200, (2, 5, 5)), (201, (2, 5, 6)), (202, (2, 5, 6)), (100, (2, 5, 5)), (103, (1, 5, 5))):
        c = column[index_of(compiled, body)]
        assert (int(column_side[c]), int(hexes[c][0]), int(hexes[c][1])) == expected
    kinds = compiled.strings["column_kind"]
    c56 = column[index_of(compiled, 201)]
    assert kinds[compiled["column_kind"][c56]] == "pale"        # R7p and R8y tie; the tie is counted
    assert compiled.counts["column_kind_conflicts"] == 1
    assert compiled.counts["columns"] == {"left": 1, "right": 2}
