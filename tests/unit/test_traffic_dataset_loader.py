from rs.runtime.offline.traffic_dataset import load_traffic_instances


def test_current_trace_zip_loader(monkeypatch):
    # The real package path is injected only in the local evidence run; unit
    # repositories without the package skip this integration-style check.
    import os
    path = os.environ.get("ROUTERSENSE_TEST_TRACE_ZIP")
    if not path:
        return
    records = load_traffic_instances(path, split="validation", world_sizes=(8,))
    assert len(records) == 256
    assert all(record.world_size == 8 for record in records)
