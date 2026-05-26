"""Tests for vector-index export/import on the embedded Python client."""

from samyama import SamyamaClient


def test_vector_index_export_import_round_trip(tmp_path):
    """export_vectors + import_vectors preserves the HNSW index across clients."""
    vec_dir = tmp_path / "vectors"

    # Source client: build index, populate, dump.
    a = SamyamaClient.embedded()
    a.create_vector_index("Doc", "embedding", 4, "cosine")
    a.query("CREATE (d:Doc {title: 'Alpha'})")
    a.query("CREATE (d:Doc {title: 'Beta'})")

    rows = a.query_readonly("MATCH (d:Doc) RETURN id(d) ORDER BY d.title")
    nid_alpha = rows.records[0][0]
    nid_beta = rows.records[1][0]
    a.add_vector("Doc", "embedding", nid_alpha, [1.0, 0.0, 0.0, 0.0])
    a.add_vector("Doc", "embedding", nid_beta, [0.0, 1.0, 0.0, 0.0])

    r1 = a.vector_search("Doc", "embedding", [1.0, 0.1, 0.0, 0.0], 2)
    assert len(r1) == 2
    assert r1[0][0] == nid_alpha  # sanity: Alpha is closer

    a.export_vectors(str(vec_dir))
    assert (vec_dir / "metadata.json").exists()

    # Fresh client: re-create nodes (same allocation order → same NodeIds),
    # declare the index, then load the HNSW state from disk. No add_vector.
    #
    # The dump records NodeIds, not property identifiers, so this round-trip
    # only works because GraphStore allocates NodeIds sequentially from a
    # fresh store. The intended end-to-end flow pairs this with
    # `import_snapshot` (which restores the nodes with their original
    # NodeIds); the snapshot binding is a separate PR. This test exercises
    # the narrow `export_vectors` / `import_vectors` contract on its own.
    b = SamyamaClient.embedded()
    b.create_vector_index("Doc", "embedding", 4, "cosine")
    b.query("CREATE (d:Doc {title: 'Alpha'})")
    b.query("CREATE (d:Doc {title: 'Beta'})")
    b.import_vectors(str(vec_dir))

    r2 = b.vector_search("Doc", "embedding", [1.0, 0.1, 0.0, 0.0], 2)
    assert [hit[0] for hit in r2] == [hit[0] for hit in r1]
    # Distances are deterministic for the same query against the same index.
    for (id1, d1), (id2, d2) in zip(r1, r2):
        assert id1 == id2
        assert abs(d1 - d2) < 1e-6


def test_import_vectors_missing_dir_is_noop(tmp_path):
    """import_vectors on a non-existent directory is a no-op, not an error."""
    client = SamyamaClient.embedded()
    client.create_vector_index("Doc", "embedding", 4, "cosine")
    # Per VectorIndexManager: missing metadata.json → no-op.
    client.import_vectors(str(tmp_path / "no_such_dir"))
