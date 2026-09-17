"""
SQLite-backed persistence for CostTracker.

Stores every cost event so that spend can be queried across restarts and
processes.  The schema is append-only; no row is ever updated or deleted
during normal operation (cleanup removes old rows).

Schema
------
    cost_events(
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        provider          TEXT    NOT NULL,
        model             TEXT    NOT NULL,
        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd          REAL    NOT NULL DEFAULT 0.0,
        timestamp         REAL    NOT NULL   -- UNIX epoch (time.time())
    )

Concurrency
-----------
WAL journal mode + BEGIN IMMEDIATE give multi-reader / single-writer
semantics safe for concurrent processes without external locking, so a
file-backed store hands each thread its own connection. A ``:memory:``
database exists only inside the connection that opened it, so that store keeps
one shared connection and serializes statements on it — otherwise every thread
but the first would be writing to a database of its own.

Usage::

    from effgen.models._cost_store import SQLiteCostStore
    store = SQLiteCostStore()            # ~/.effgen/costs.sqlite
    store = SQLiteCostStore(":memory:")  # in-process tests

    store.insert(provider="cerebras", model="llama3.1-8b",
                 prompt_tokens=50, completion_tokens=20,
                 cost_usd=0.0, timestamp=time.time())

    rows = store.query_today()
    rows = store.query_since(since_timestamp)
    rows = store.query_all()

    total = store.spend_today()      # summed in SQLite, no objects built
    total = store.spend_since(since_timestamp)
    n = store.count()
    removed = store.prune(max_age_days=90)

Reading spend
-------------
``spend_*`` returns the one number a budget check needs, summed in the database
against an index on ``timestamp``, so the cost of a check follows the window it
asks about rather than the size of the table. ``query_*`` returns the rows
themselves and is what a report needs; it builds one :class:`CostEvent` per row.

Growth
------
The table gains a row per model call and nothing removes one during normal
operation. Crossing :data:`RETENTION_WARN_ROWS` logs one line naming
``effgen cost prune``; :meth:`SQLiteCostStore.prune` is the only thing that
deletes, and only when it is called.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".effgen" / "costs.sqlite"


def _default_db_path() -> Path:
    """Where cost events are stored when no path is given.

    ``EFFGEN_HOME`` relocates the whole effGen state directory, so the cost
    database follows it to ``$EFFGEN_HOME/costs.sqlite``; otherwise it lives at
    ``~/.effgen/costs.sqlite``.
    """
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "costs.sqlite"
    return _DEFAULT_DB_PATH

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cost_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    timestamp         REAL    NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_cost_events_lookup
    ON cost_events (provider, model, timestamp);
"""

#: Budget checks filter on time alone. The composite index above leads with
#: ``provider``, so SQLite cannot seek into it on a bare ``timestamp >= ?`` and
#: falls back to walking every distinct (provider, model) prefix. Every model
#: call runs a budget preflight, so that walk was paid once per call against a
#: table that grows by one row per call.
#:
#: ``cost_usd`` is carried in the index rather than only ``timestamp`` because
#: the sum below is then answered from the index alone. With a timestamp-only
#: index SQLite still has to fetch each matching row from the table to read the
#: one column it is adding up, which on a 500,000-row ledger measured ~370 ms
#: for the 30-day window against ~9 ms covered.
_CREATE_TIME_INDEX = """
CREATE INDEX IF NOT EXISTS idx_cost_events_timestamp
    ON cost_events (timestamp, cost_usd);
"""

#: Spend for a period, summed in SQLite. The budget check only ever wanted the
#: total, but read it through :data:`_QUERY_SINCE` and added the rows up in
#: Python, so a preflight built one :class:`CostEvent` per row in the window and
#: discarded all of them. On a 500,000-row ledger that measured ~894 ms per
#: call, and because every call made the same query, concurrent agents queued
#: behind it: throughput at 16 agents was worse than at one.
_SUM_SINCE = """
SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_events WHERE timestamp >= ?;
"""

_COUNT_ALL = """
SELECT COUNT(*) FROM cost_events;
"""

_COUNT_SINCE = """
SELECT COUNT(*) FROM cost_events WHERE timestamp >= ?;
"""

_INSERT = """
INSERT INTO cost_events (provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)
VALUES (?, ?, ?, ?, ?, ?);
"""

_QUERY_SINCE = """
SELECT provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp
FROM cost_events
WHERE timestamp >= ?
ORDER BY timestamp ASC;
"""

_QUERY_ALL = """
SELECT provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp
FROM cost_events
ORDER BY timestamp ASC;
"""

_DELETE_OLD = """
DELETE FROM cost_events WHERE timestamp < ?;
"""

_DELETE_KEEP_NEWEST = """
DELETE FROM cost_events WHERE id NOT IN (
    SELECT id FROM cost_events ORDER BY timestamp DESC LIMIT ?
);
"""

#: The ledger gains a row per model call and nothing removes one, so a
#: long-lived process accumulates without bound. These are the documented
#: ceiling: crossing :data:`RETENTION_WARN_ROWS` prints one line naming
#: ``effgen cost prune``, and ``prune`` with no bound keeps
#: :data:`RETENTION_MAX_AGE_DAYS` of history. Neither deletes anything on its
#: own — the ledger is the user's spend record, and `effgen cost by-provider`
#: reports it over the store's whole lifetime.
RETENTION_WARN_ROWS = 250_000
RETENTION_MAX_AGE_DAYS = 90.0

#: SQLite page cache per connection, in KiB. A file-backed store opens one
#: connection per thread that uses it, and SQLite's default gives each 2 MiB, so
#: a process running 32 agents held up to 64 MiB of cache for a table whose
#: budget query is answered from an index; measured here, resident memory rose
#: with the ledger until every connection's cache was full. The operating
#: system's own file cache still serves the pages; this bounds only what each
#: connection keeps for itself.
CONNECTION_CACHE_KIB = 256

#: How long a caller whose event is being written by another caller's
#: transaction waits before looking again. A bound on one wait, not a deadline:
#: the caller keeps waiting until its event is written or dropped.
PENDING_WAIT_S = 30.0


class _PendingRows(threading.Condition):
    """The rows recorded but not yet written, and the lock that guards them.

    A condition rather than a lock because callers wait on it: a caller whose
    event is being written by another caller's transaction waits here, not on
    the database, and is woken when that transaction ends.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[tuple[str, str, int, int, float, float]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append_row(self, row: tuple[str, str, int, int, float, float]) -> None:
        self._rows.append(row)

    def take(self) -> list[tuple[str, str, int, int, float, float]]:
        """Return everything waiting and leave the list empty."""
        rows, self._rows = self._rows, []
        return rows


@dataclass
class CostEvent:
    """One recorded cost event row."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float


class SQLiteCostStore:
    """Thread-safe, cross-process SQLite store for cost events.

    Args:
        db_path: Path to the SQLite file.  Use ``":memory:"`` for tests.
                 Defaults to ``$EFFGEN_HOME/costs.sqlite`` when ``EFFGEN_HOME``
                 is set, else ``~/.effgen/costs.sqlite``. ``EFFGEN_COST_DB``
                 overrides both.
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            # Allow tests / sandboxes to redirect persistence away from the
            # user's real ~/.effgen/costs.sqlite via EFFGEN_COST_DB.
            env_path = os.environ.get("EFFGEN_COST_DB")
            db_path = env_path if env_path else _default_db_path()
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # An in-memory database belongs to the connection that opened it, so a
        # per-thread connection would give every thread its own empty database
        # and every call recorded off the creating thread would be lost. One
        # connection, shared, with a lock around each statement.
        self._shared_lock = threading.Lock() if self._path == ":memory:" else None
        self._shared_conn: sqlite3.Connection | None = None
        #: Rows currently stored, counted once on the first insert and then
        #: tracked in process. Counting per insert would put a second query on
        #: the write path to answer a question that only changes by one.
        self._rows: int | None = None
        self._warned_retention = False
        #: Events recorded and not yet written. A model call records one and
        #: returns once it is written; whichever caller finds the write token
        #: free writes everything waiting in one transaction, and the others
        #: wait for that transaction rather than each queueing inside SQLite for
        #: a transaction of its own. SQLite allows one writer at a time.
        self._pending = _PendingRows()
        #: One writer inside SQLite at a time: the caller writing the current
        #: batch. Nobody waits on it; a caller that finds it taken waits on
        #: ``_pending`` for the batch in progress to end.
        self._write_lock = threading.Lock()
        #: Under ``_pending``: events recorded, events written or dropped, and
        #: batches finished. An event is done once ``_settled`` reaches its
        #: sequence number.
        self._recorded = 0
        self._settled = 0
        self._batches_done = 0
        #: Events written, transactions they were written in, and how often a
        #: caller's event was written by another caller's transaction.
        #: ``events / batches`` is how many calls one transaction covers.
        self.write_stats: dict[str, int] = {"events": 0, "batches": 0, "waited_for_write": 0}
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA cache_size=-{int(CONNECTION_CACHE_KIB)};")
        if self._path == ":memory:":
            # An in-memory database is created empty along with its connection,
            # so the schema belongs to opening one rather than to constructing
            # the store: a store reopened after close() would otherwise have
            # nothing to write to and drop every call recorded on it.
            with conn:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
                self._create_time_index(conn)
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._shared_lock is not None:
            with self._shared_lock:
                if self._shared_conn is None:
                    self._shared_conn = self._open()
                return self._shared_conn
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._open()
        return cast(sqlite3.Connection, self._local.conn)

    @contextmanager
    def _exclusive(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection, serialized when it is shared across threads.

        A file-backed store gives each thread its own connection and SQLite's
        own locking keeps writers apart, so there is nothing to serialize. The
        single in-memory connection is shared, and two threads issuing
        statements on one connection interleave, so that case takes a lock.
        """
        conn = self._conn()
        if self._shared_lock is None:
            yield conn
            return
        with self._shared_lock:
            yield conn

    @staticmethod
    def _create_time_index(conn: sqlite3.Connection) -> None:
        """Add the timestamp index, tolerating a store that cannot be written.

        The index is what makes a budget query proportional to its window
        instead of to the whole ledger, but a read-only file, a ledger on a
        read-only mount, or a database another process is holding must still be
        *readable*: a cost ledger that refuses to open would take the model call
        down with it. So a failure here degrades the query plan and nothing
        else, and is reported at debug level rather than raised.
        """
        try:
            conn.execute(_CREATE_TIME_INDEX)
        except sqlite3.Error:
            logger.debug("Could not create %s; budget queries will be slower",
                         "idx_cost_events_timestamp", exc_info=True)

    def _init_schema(self) -> None:
        with self._exclusive() as conn, conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            self._create_time_index(conn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        timestamp: float | None = None,
    ) -> None:
        """Insert one cost event atomically, and return once it is written.

        Calls recorded at the same time are written together: the caller that
        finds no write in progress writes every event waiting, its own
        included, in one transaction, and a caller that finds one in progress
        waits for it to end rather than queueing inside SQLite for a
        transaction of its own. An event is in the file when this returns, so
        a process that ends at any moment afterwards — an exit, a signal, a
        crash — has not lost it.

        Args:
            provider: The provider that served the call.
            model: The model id the call used.
            prompt_tokens: Input tokens the call consumed.
            completion_tokens: Output tokens the call produced.
            cost_usd: What the call cost in US dollars.
            timestamp: Unix time of the call, defaulting to now.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._pending:
            self._pending.append_row(
                (provider, model, prompt_tokens, completion_tokens, cost_usd, ts)
            )
            self._recorded += 1
            mine = self._recorded
        self._write_through(mine)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _write_through(self, seq: int, timeout: float | None = None) -> bool:
        """Return once every event up to *seq* is written (or dropped and said so).

        The caller writes the batch itself when no write is in progress. When
        one is, it waits for that batch to end and looks again: its event is
        either in that batch or in the next one, which it or another waiting
        caller writes. Returns ``False`` only when *timeout* ran out first.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        waited = False
        while True:
            with self._pending:
                if self._settled >= seq:
                    if waited:
                        self.write_stats["waited_for_write"] += 1
                    return True
                batches_seen = self._batches_done
            if self._write_lock.acquire(blocking=False):
                try:
                    self._drain()
                finally:
                    self._write_lock.release()
                    with self._pending:
                        self._batches_done += 1
                        self._pending.notify_all()
                continue
            waited = True
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            wait_s = PENDING_WAIT_S if remaining is None else min(PENDING_WAIT_S, remaining)
            with self._pending:
                # Woken when the batch in progress ends; checked under the same
                # lock that batch reports under, so its end is never missed.
                if self._settled < seq and self._batches_done == batches_seen:
                    self._pending.wait(timeout=wait_s)

    def flush(self, timeout: float | None = None) -> int:
        """Write every recorded event that has not reached the file yet.

        :meth:`insert` returns only once its event is written, so this finds
        work only while other threads are inside :meth:`insert`.

        Args:
            timeout: How long to wait for a write in progress, in seconds.
                ``None`` waits until it ends.

        Returns:
            How many events reached the file while it waited, or ``0`` when the
            wait ran out.
        """
        with self._pending:
            target = self._recorded
            before = self._settled
        if not self._write_through(target, timeout=timeout):
            logger.debug("cost ledger: a write in progress did not end in %.1fs", timeout)
            return 0
        with self._pending:
            return max(0, self._settled - before)

    def _drain(self) -> int:
        """Write every waiting event in one transaction. The caller holds the write lock.

        A batch that cannot be written is dropped and the number of events in
        it is logged: a recorded call is never the reason a run fails.
        """
        with self._pending:
            batch = self._pending.take()
        if not batch:
            return 0
        written = 0
        try:
            with self._exclusive() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    conn.executemany(_INSERT, batch)
                    conn.execute("COMMIT;")
                except Exception:
                    conn.execute("ROLLBACK;")
                    raise
            written = len(batch)
        except Exception as exc:
            logger.warning("Cost ledger write failed; %d event(s) not recorded: %s",
                           len(batch), exc)
        finally:
            with self._pending:
                self._settled += len(batch)
                if written:
                    self.write_stats["events"] += written
                    self.write_stats["batches"] += 1
        if written:
            self._note_insert(written)
            logger.debug("cost ledger: wrote %d event(s) in one transaction", written)
        return written

    def _note_insert(self, added: int = 1) -> None:
        """Track the row count and say once when the ledger crosses its ceiling.

        The count is read from the database once, on the first insert of this
        store, and incremented from then on: the write path already knows it
        added exactly one row, so asking the database again per call would put a
        second query on it to learn something it could have counted.

        The message is emitted once per store. It says what to run, and it does
        not prune: a spend record is the user's to keep or drop.
        """
        try:
            if self._rows is None:
                self._rows = self._count_rows()
            self._rows += added
            if self._warned_retention or self._rows < RETENTION_WARN_ROWS:
                return
            self._warned_retention = True
            logger.warning(
                "effGen cost ledger has %d events (%s). Budget checks stay fast, "
                "but the file only grows; run 'effgen cost prune' to keep the "
                "last %d days.",
                self._rows, self._path, int(RETENTION_MAX_AGE_DAYS),
            )
        except sqlite3.Error:
            # Bookkeeping must never be the reason a recorded call fails.
            logger.debug("Could not track cost-ledger size", exc_info=True)

    def query_since(self, since: float) -> list[CostEvent]:
        """Return all events with timestamp >= *since*."""
        self.flush()
        with self._exclusive() as conn:
            rows = conn.execute(_QUERY_SINCE, (since,)).fetchall()
        return [CostEvent(*row) for row in rows]

    def spend_since(self, since: float) -> float:
        """Return total USD spend with ``timestamp >= since``, summed in SQLite.

        The budget check wants one number. Reading it through
        :meth:`query_since` builds a :class:`CostEvent` for every row in the
        window only to add up one field and throw the objects away, so the cost
        of a check grew with the ledger rather than with the window. This runs
        the sum in the database against an index on ``timestamp``.
        """
        self.flush()
        with self._exclusive() as conn:
            row = conn.execute(_SUM_SINCE, (since,)).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def spend_today(self) -> float:
        """Total USD spend over the last 24 hours (rolling day)."""
        return self.spend_since(time.time() - 86400.0)

    def spend_week(self) -> float:
        """Total USD spend over the last 7 days."""
        return self.spend_since(time.time() - 7 * 86400.0)

    def spend_month(self) -> float:
        """Total USD spend over the last 30 days."""
        return self.spend_since(time.time() - 30 * 86400.0)

    def count(self) -> int:
        """Number of events currently stored."""
        self.flush()
        return self._count_rows()

    def _count_rows(self) -> int:
        """Rows in the table, without writing anything waiting first.

        The write path itself counts through here: it already holds the write
        token, so asking :meth:`count` — which flushes — would be asking for a
        token it is holding.
        """
        with self._exclusive() as conn:
            row = conn.execute(_COUNT_ALL).fetchone()
        return int(row[0]) if row else 0

    def count_since(self, since: float) -> int:
        """Number of events with ``timestamp >= since``, counted in SQLite."""
        self.flush()
        with self._exclusive() as conn:
            row = conn.execute(_COUNT_SINCE, (since,)).fetchone()
        return int(row[0]) if row else 0

    def prune(self, *, max_age_days: float | None = None,
              keep_rows: int | None = None) -> int:
        """Delete old events and return how many rows went.

        Exactly one bound is applied per call. ``max_age_days`` drops everything
        older than that many days; ``keep_rows`` keeps the newest *keep_rows*
        events and drops the rest. With neither, :data:`RETENTION_MAX_AGE_DAYS`
        applies, which is the ceiling this store documents.

        Pruning is never automatic. The ledger is the user's own spend record
        and `effgen cost by-provider` reports it over the store's whole
        lifetime, so rows are removed when someone asks and not before.
        """
        if max_age_days is not None and keep_rows is not None:
            raise ValueError(
                "prune() was given both max_age_days and keep_rows. "
                "Pass one bound per call."
            )
        self.flush()
        with self._exclusive() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                if keep_rows is not None:
                    if keep_rows < 0:
                        raise ValueError(
                            "keep_rows is negative. Pass 0 or more to say how "
                            "many of the newest events to keep."
                        )
                    cursor = conn.execute(_DELETE_KEEP_NEWEST, (keep_rows,))
                else:
                    days = (RETENTION_MAX_AGE_DAYS if max_age_days is None
                            else float(max_age_days))
                    cursor = conn.execute(_DELETE_OLD, (time.time() - days * 86400.0,))
                count = cursor.rowcount
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
        if count:
            self._rows = None
        return count

    def query_today(self) -> list[CostEvent]:
        """Return events from the last 24 hours (rolling day)."""
        since = time.time() - 86400.0
        return self.query_since(since)

    def query_week(self) -> list[CostEvent]:
        """Return events from the last 7 days."""
        since = time.time() - 7 * 86400.0
        return self.query_since(since)

    def query_month(self) -> list[CostEvent]:
        """Return events from the last 30 days."""
        since = time.time() - 30 * 86400.0
        return self.query_since(since)

    def query_all(self) -> list[CostEvent]:
        """Return all stored events (lifetime)."""
        self.flush()
        with self._exclusive() as conn:
            rows = conn.execute(_QUERY_ALL).fetchall()
        return [CostEvent(*row) for row in rows]

    def cleanup(self, max_age_seconds: float) -> int:
        """Delete events older than *max_age_seconds*.  Returns rows deleted."""
        cutoff = time.time() - max_age_seconds
        self.flush()
        with self._exclusive() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(_DELETE_OLD, (cutoff,))
                count = cursor.rowcount
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
        if count:
            self._rows = None       # re-count on the next insert
        return count

    def close(self) -> None:
        """Close this thread's connection, or the shared in-memory one.

        An event being recorded on another thread at the same moment is
        written first, so closing a store never loses a call it accepted.
        """
        self.flush()
        if self._shared_lock is not None:
            with self._shared_lock:
                if self._shared_conn is not None:
                    self._shared_conn.close()
                    self._shared_conn = None
            return
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
