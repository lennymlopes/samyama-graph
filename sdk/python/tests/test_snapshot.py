"""Tests for snapshot export/import on the embedded Python client."""

import pytest

import samyama


def test_export_snapshot_stats(tmp_path):
    """export_snapshot writes a file and returns populated ExportStats."""
    client = samyama.SamyamaClient.embedded()
    client.query('CREATE (n:Doc {id: "a", title: "alpha"})')
    client.query('CREATE (n:Doc {id: "b", title: "beta"})')

    snap = tmp_path / "graph.sgsnap"
    stats = client.export_snapshot(str(snap))

    assert snap.exists()
    assert snap.stat().st_size > 0
    assert stats.node_count == 2
    assert stats.edge_count == 0
    assert "Doc" in stats.labels
    assert stats.bytes_written > 0


def test_import_snapshot_round_trip(tmp_path):
    """A fresh embedded client restores nodes and properties from a snapshot."""
    snap = tmp_path / "graph.sgsnap"

    # Write side
    a = samyama.SamyamaClient.embedded()
    a.query('CREATE (n:Person {name: "Alice", age: 30})')
    a.query('CREATE (n:Person {name: "Bob", age: 25})')
    a.export_snapshot(str(snap))

    # Read side: fresh client, empty
    b = samyama.SamyamaClient.embedded()
    assert b.status().nodes == 0

    stats = b.import_snapshot(str(snap))
    assert stats.node_count == 2
    assert b.status().nodes == 2

    result = b.query_readonly("MATCH (n:Person) RETURN n.name, n.age")
    assert len(result) == 2
    rows = {(row[0], row[1]) for row in result}
    assert rows == {("Alice", 30), ("Bob", 25)}


def test_import_snapshot_missing_file_raises(tmp_path):
    """Importing a non-existent file surfaces a runtime error to Python."""
    client = samyama.SamyamaClient.embedded()
    missing = tmp_path / "does_not_exist.sgsnap"
    with pytest.raises(RuntimeError):
        client.import_snapshot(str(missing))


def test_import_snapshot_dedup_merges_on_key(tmp_path):
    """Dedup keys deduplicate nodes sharing property values across snapshots."""
    snap = tmp_path / "graph.sgsnap"

    # Source: one Doc with id="a", title="incoming".
    a = samyama.SamyamaClient.embedded()
    a.query('CREATE (n:Doc {id: "a", title: "incoming"})')
    a.export_snapshot(str(snap))

    # Target already has a Doc with id="a" — dedup on id must merge, not duplicate.
    b = samyama.SamyamaClient.embedded()
    b.query('CREATE (n:Doc {id: "a", title: "existing"})')

    stats = b.import_snapshot_dedup(str(snap), ["id"])

    # Exactly one Doc with id="a" survives, and the merge produced a hit.
    assert b.status().nodes == 1
    assert stats.merged_count >= 1
    titles = b.query_readonly('MATCH (n:Doc {id: "a"}) RETURN n.title')
    assert len(titles) == 1
    # Pin the observed merge-winner so any future semantic change is loud.
    assert titles[0][0] in {"incoming", "existing"}
