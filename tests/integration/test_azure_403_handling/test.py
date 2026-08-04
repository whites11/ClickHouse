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
import time

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


def _create_table(name, wide=True, stop_merges=False, policy="azure_policy"):
    node.query(f"DROP TABLE IF EXISTS {name} SYNC")
    part_setting = (
        "min_bytes_for_wide_part = 0" if wide else "min_bytes_for_wide_part = 1073741824"
    )
    node.query(
        f"""
        CREATE TABLE {name} (k UInt64, v String)
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/{name}', 'r1')
        ORDER BY k
        SETTINGS storage_policy = '{policy}', {part_setting}
        """
    )
    # Keep the three inserted parts distinct so a later OPTIMIZE has real work to merge — the merge
    # test needs a merge that actually reaches the read (and the reportBroken() decision point).
    if stop_merges:
        node.query(f"SYSTEM STOP MERGES {name}")
    for i in range(3):
        node.query(
            f"INSERT INTO {name} SELECT number + {i * 100}, toString(number) FROM numbers(100)"
        )
    assert node.query(f"SELECT count() FROM {name}").strip() == "300"
    node.query("SYSTEM DROP FILESYSTEM CACHE")
    node.query("SYSTEM DROP MARK CACHE")


def _wait_for_merge_failure(table, expected_in_err, timeout=90):
    # reportBroken() is decided only after a merge's read exhausts its retry budget and the final
    # exception reaches MergeTreeReader*. A retryable permanent error reschedules the merge forever,
    # so the only terminal signal is the failure recorded on the replication queue. Block (with the
    # failpoint still enabled) until the injected error surfaces there; only then is a no-broken-part
    # assertion meaningful.
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        last = node.query(
            f"SELECT last_exception FROM system.replication_queue "
            f"WHERE database = currentDatabase() AND table = '{table}'"
        )
        if expected_in_err in last:
            return
        time.sleep(0.5)
    raise AssertionError(
        f"merge for {table} never recorded a failure containing {expected_in_err!r}; "
        f"last_exception=\n{last}"
    )


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


def test_sdk_retry_isolates_forbidden_on_direct_call(started_cluster):
    # Isolation test for the SDK retry surface: AzureBlobStorageCommon.cpp adds Forbidden to
    # retry_options.StatusCodes so the SDK RetryPolicy retries a returned 403. Setting up an INSERT into
    # a table function issues direct GetProperties() SDK calls (the container check, then exists() for
    # the blob — BlobContainerClient/BlobClient::GetProperties via AzureObjectStorage.cpp and
    # checkAndGetNewFileOnInsertIfNeeded in Utils.cpp) that have NO ClickHouse-level retry loop, so a
    # one-shot 403 on the first of them can be absorbed only by the SDK retry — i.e. only by that line.
    # With the one-shot armed the INSERT therefore succeeds ONLY because the SDK retried the 403; remove
    # the StatusCodes.insert(Forbidden) line and this test fails with "403 Forbidden" out of
    # GetProperties, whereas the read/write tests keep passing via their own ClickHouse retry loops —
    # exactly the gap the reviewer flagged. Runs early, while the node is otherwise Azure-quiet, so the
    # one-shot lands on this INSERT's first metadata call.
    endpoint = started_cluster.env_variables["AZURITE_STORAGE_ACCOUNT_URL"]

    node.query("SYSTEM ENABLE FAILPOINT azure_inject_forbidden_response_once")
    try:
        node.query(
            f"""
            INSERT INTO TABLE FUNCTION azureBlobStorage(
                '{endpoint}', '{CONTAINER}', 'b1_direct_probe.csv',
                '{AZURITE_ACCOUNT}', '{AZURITE_KEY}', 'CSV', 'auto', 'k UInt64')
            VALUES (1)
            """
        )
    finally:
        node.query("SYSTEM DISABLE FAILPOINT azure_inject_forbidden_response_once")


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
    # The load-bearing check: the healthy part must never be reported broken. (There is no assertion on
    # POTENTIALLY_BROKEN_DATA_PART / code 740 — it is thrown only in the private build, so asserting its
    # absence from `err` would be vacuous in OSS. BROKEN_PART_LOG, emitted by
    # ReplicatedMergeTreePartCheckThread, is the OSS-observable signal that reportBroken() was taken.)
    assert not node.contains_in_log(BROKEN_PART_LOG)
    assert node.contains_in_log(RETRY_LOG), "the read retry budget was never used"


@pytest.mark.parametrize("kind", list(ERROR_KINDS))
def test_permanent_error_at_merge_does_not_mark_part_broken(started_cluster, kind):
    # A permanent error hit by a background merge must be retried (the merge just fails and is
    # rescheduled), never attributed to a broken part. OPTIMIZE is async (alter_sync=0): a retryable
    # failure makes the merge retry indefinitely, so a synchronous OPTIMIZE would block until the
    # query timeout instead of returning.
    _, perm_fp, expected_in_err = ERROR_KINDS[kind]
    table = f"t_merge_{kind}"
    _create_table(table, stop_merges=True)

    node.query(f"SYSTEM ENABLE FAILPOINT {perm_fp}")
    try:
        node.query(f"SYSTEM START MERGES {table}")
        node.query(f"OPTIMIZE TABLE {table} FINAL SETTINGS alter_sync = 0")
        # Keep the failpoint enabled until the merge reaches its terminal failure path. Checking at the
        # first RETRY_LOG (as before) fires before the read budget is exhausted and reportBroken() is
        # decided, so a false reportBroken() on the final attempt would slip through — and the old
        # SELECT count() == 300 does not catch it either, since a bad reportBroken() removes and
        # refetches the healthy part and still leaves 300 rows. Instead wait for the injected error to
        # surface on the replication queue (the merge failed retryably), then assert no broken-part log.
        _wait_for_merge_failure(table, expected_in_err, timeout=90)
        assert not node.contains_in_log(BROKEN_PART_LOG)
    finally:
        node.query(f"SYSTEM DISABLE FAILPOINT {perm_fp}")


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


def test_transient_forbidden_on_write_succeeds(started_cluster):
    # The write-side counterpart to test_transient_error_read_succeeds, and the only case that pins the
    # CH-level write contract the fix widened: WriteBufferFromAzureBlobStorage::execWithRetry classifying
    # a 403 as retryable (isRetryableAzureException). It runs on azure_policy_nosdk, whose max_tries=0
    # turns the SDK RetryPolicy off, because with the SDK retry on (the default azure_disk, max_tries=2)
    # a *returned* one-shot 403 is retried by the SDK one layer below execWithRetry — the upload then
    # succeeds even if the CH write loop stopped retrying 403, so that config cannot detect the
    # regression the reviewer flagged. test_permanent_forbidden_on_write_fails has the same blind spot:
    # its final error text is "403" whether the loop retried-then-gave-up or failed fast.
    #
    # With the SDK retry off, the returned one-shot 403 reaches execWithRetry, which must retry it for
    # the INSERT to succeed; the "Write at attempt" debug line is emitted only by that CH-level retry
    # (WriteBufferFromAzureBlobStorage.cpp:132), so it proves the write loop — not the SDK — recovered.
    # stop_merges keeps the armed INSERT's part upload the only Azure traffic in flight, so the global
    # one-shot lands on the upload PUT rather than a stray background merge.
    _create_table("t_write_transient", stop_merges=True, policy="azure_policy_nosdk")

    node.query("SYSTEM ENABLE FAILPOINT azure_inject_forbidden_response_once")
    try:
        node.query(
            "INSERT INTO t_write_transient SELECT number + 300, toString(number) FROM numbers(100)"
        )
    finally:
        node.query("SYSTEM DISABLE FAILPOINT azure_inject_forbidden_response_once")

    assert node.query("SELECT count() FROM t_write_transient").strip() == "400"
    assert node.contains_in_log(
        "Write at attempt"
    ), "the CH-level write retry loop was never exercised"
    assert not node.contains_in_log(BROKEN_PART_LOG)
