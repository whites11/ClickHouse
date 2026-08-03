#!/usr/bin/env python3
"""Azure transient-vs-permanent error handling on the MergeTree read/merge path.

Covers #106298, #110724 and private #53656: a transient Azure 403, credential failure or connect
timeout must be retried and must never be attributed to a broken data part; a permanent one must
fail with the real Azure error and still not accuse the part. A genuinely non-retryable error must
still mark the part broken.

The table is ReplicatedMergeTree on purpose: plain MergeTree's broken_part_callback is a no-op
(MergeTreeData.h:526), so reportBroken() would have no observable effect and the tests could not
distinguish fixed from broken. On Replicated, reportBroken() reaches
ReplicatedMergeTreePartCheckThread and logs BROKEN_PART_LOG.
"""
import os

import pytest

from helpers.cluster import ClickHouseCluster

AZURITE_ACCOUNT = "devstoreaccount1"
AZURITE_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
CONTAINER = "cont"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

cluster = ClickHouseCluster(__file__)
node = cluster.add_instance(
    "node",
    main_configs=[os.path.join(SCRIPT_DIR, "configs", "azure_disk.xml")],
    with_azurite=True,
    with_zookeeper=True,
)

# kind -> (one-shot failpoint, permanent failpoint, text expected in the permanent error)
ERROR_KINDS = {
    "forbidden": (
        "azure_inject_forbidden_response_once",
        "azure_inject_forbidden_response",
        "403",
    ),
    "auth": (
        "azure_inject_auth_failure_on_request_once",
        "azure_inject_auth_failure_on_request",
        "AuthenticationException",
    ),
    "timeout": (
        "azure_inject_poco_timeout_once",
        "azure_inject_poco_timeout",
        "TransportException",
    ),
}

ALL_FAILPOINTS = [fp for triple in ERROR_KINDS.values() for fp in triple[:2]] + [
    "azure_inject_bad_request"
]

# Emitted by ReplicatedMergeTreePartCheckThread.cpp:439 once reportBroken() is taken.
BROKEN_PART_LOG = "looks broken. Removing it and will try to fetch"
# Emitted per failed attempt by the Azure read/download retry loop in ReadBufferFromAzureBlobStorage
# (a part read goes through the "Download" variant; keep the match broad enough for both).
RETRY_LOG = "Exception caught during Azure"


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


@pytest.fixture(autouse=True, scope="function")
def clean_state(started_cluster):
    def _disable_all():
        for fp in ALL_FAILPOINTS:
            try:
                node.query(f"SYSTEM DISABLE FAILPOINT {fp}")
            except Exception:
                pass

    _disable_all()
    node.rotate_logs()
    yield
    _disable_all()


def _create_table(name, wide=True):
    node.query(f"DROP TABLE IF EXISTS {name} SYNC")
    part_setting = (
        "min_bytes_for_wide_part = 0" if wide else "min_bytes_for_wide_part = 1073741824"
    )
    node.query(
        f"""
        CREATE TABLE {name} (k UInt64, v String)
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/{name}', 'r1')
        ORDER BY k
        SETTINGS storage_policy = 'azure_policy', {part_setting}
        """
    )
    for i in range(3):
        node.query(
            f"INSERT INTO {name} SELECT number + {i * 100}, toString(number) FROM numbers(100)"
        )
    assert node.query(f"SELECT count() FROM {name}").strip() == "300"
    node.query("SYSTEM DROP FILESYSTEM CACHE")
    node.query("SYSTEM DROP MARK CACHE")


def test_sanity_check(started_cluster):
    endpoint = started_cluster.env_variables["AZURITE_STORAGE_ACCOUNT_URL"]
    node.query(
        f"""
        CREATE TABLE t_sanity (k UInt64, v String)
        ENGINE = AzureBlobStorage('{endpoint}', '{CONTAINER}', 'sanity.csv',
                                  '{AZURITE_ACCOUNT}', '{AZURITE_KEY}', 'CSV')
        """
    )
    node.query("INSERT INTO t_sanity VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    assert node.query("SELECT count() FROM t_sanity").strip() == "3"


@pytest.mark.parametrize("kind", list(ERROR_KINDS))
def test_transient_error_read_succeeds(started_cluster, kind):
    # A single transient failure must be absorbed by the retry budget: the read still returns data,
    # and the part is never reported broken. No assertion on RETRY_LOG here: with 403 in the SDK
    # retryable set the SDK may absorb the one-shot failure internally, below the ClickHouse loop.
    once_fp, _, _ = ERROR_KINDS[kind]
    table = f"t_transient_{kind}"
    _create_table(table)

    node.query(f"SYSTEM ENABLE FAILPOINT {once_fp}")
    try:
        assert node.query(f"SELECT sum(k) FROM {table}").strip() == "44850"
    finally:
        node.query(f"SYSTEM DISABLE FAILPOINT {once_fp}")

    assert not node.contains_in_log(BROKEN_PART_LOG)


@pytest.mark.parametrize("kind", list(ERROR_KINDS))
def test_permanent_error_read_fails_without_accusing_part(started_cluster, kind):
    # A never-clearing error must fail the read with the real Azure error, exhaust the ClickHouse
    # retry budget, and still never report the (healthy) part as broken.
    _, perm_fp, expected_in_err = ERROR_KINDS[kind]
    table = f"t_permanent_{kind}"
    _create_table(table)

    node.query(f"SYSTEM ENABLE FAILPOINT {perm_fp}")
    try:
        err = node.query_and_get_error(f"SELECT sum(k) FROM {table}")
    finally:
        node.query(f"SYSTEM DISABLE FAILPOINT {perm_fp}")

    assert expected_in_err in err, f"expected {expected_in_err} in error, got:\n{err}"
    assert "POTENTIALLY_BROKEN_DATA_PART" not in err
    assert not node.contains_in_log(BROKEN_PART_LOG)
    assert node.contains_in_log(RETRY_LOG), "the read retry budget was never used"


@pytest.mark.parametrize("kind", list(ERROR_KINDS))
def test_permanent_error_at_merge_does_not_mark_part_broken(started_cluster, kind):
    # A permanent error hit by a background merge must be retried (the merge just fails and is
    # rescheduled), never attributed to a broken part. OPTIMIZE is async (alter_sync=0): a retryable
    # failure makes the merge retry indefinitely, so a synchronous OPTIMIZE would block until the
    # query timeout instead of returning.
    _, perm_fp, _ = ERROR_KINDS[kind]
    table = f"t_merge_{kind}"
    _create_table(table)

    node.query(f"SYSTEM ENABLE FAILPOINT {perm_fp}")
    try:
        node.query(f"OPTIMIZE TABLE {table} FINAL SETTINGS alter_sync = 0")
        # Wait until the background merge has actually attempted the read and entered the retry loop,
        # so the no-broken-part check below is meaningful and not evaluated before anything happened.
        node.wait_for_log_line(RETRY_LOG, timeout=60)
        assert not node.contains_in_log(BROKEN_PART_LOG)
    finally:
        node.query(f"SYSTEM DISABLE FAILPOINT {perm_fp}")

    assert node.query(f"SELECT count() FROM {table}").strip() == "300"


def test_transient_forbidden_compact_part_read_succeeds(started_cluster):
    # The reportBroken() call sites differ per reader; #106298's stack is the compact reader
    # (MergeTreeReaderCompactSingleBuffer.cpp:114,147), so cover a compact part too.
    once_fp, _, _ = ERROR_KINDS["forbidden"]
    _create_table("t_compact", wide=False)

    node.query(f"SYSTEM ENABLE FAILPOINT {once_fp}")
    try:
        assert node.query("SELECT sum(k) FROM t_compact").strip() == "44850"
    finally:
        node.query(f"SYSTEM DISABLE FAILPOINT {once_fp}")

    assert not node.contains_in_log(BROKEN_PART_LOG)


def test_non_retryable_error_still_marks_part_broken(started_cluster):
    # Negative control: the fix must NOT make every Azure error retryable. A non-retryable HTTP
    # error must still take the reportBroken() path, otherwise genuine corruption goes unnoticed.
    _create_table("t_negative")

    node.query("SYSTEM ENABLE FAILPOINT azure_inject_bad_request")
    try:
        node.query_and_get_error("SELECT sum(k) FROM t_negative")
        node.wait_for_log_line(BROKEN_PART_LOG, timeout=60)
    finally:
        node.query("SYSTEM DISABLE FAILPOINT azure_inject_bad_request")


def test_permanent_forbidden_on_write_fails(started_cluster):
    # 403 is now retryable on the write path too. A permanent one must still fail (bounded budget),
    # not hang.
    _create_table("t_write")

    node.query("SYSTEM ENABLE FAILPOINT azure_inject_forbidden_response")
    try:
        err = node.query_and_get_error(
            "INSERT INTO t_write SELECT number, toString(number) FROM numbers(100)"
        )
        assert "403" in err or "Forbidden" in err
    finally:
        node.query("SYSTEM DISABLE FAILPOINT azure_inject_forbidden_response")
