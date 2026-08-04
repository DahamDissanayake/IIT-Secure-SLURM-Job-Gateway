# Audit Daemon Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop one slow request (a Resend mail send, up to 20s; a
`systemctl`/`journalctl` service-status check, up to 15s combined) from
freezing every other user's login, job submission, file access, and audit
logging — the daemon currently handles connections one at a time, inline,
on a single thread.

**Architecture:** `main()`'s accept loop submits each connection to a
bounded `ThreadPoolExecutor` instead of calling `_handle_connection`
inline. Each worker thread opens its own short-lived SQLite connections
(instead of sharing the daemon's long-lived ones) — safe with no new
locking code because `PRAGMA journal_mode=WAL` is already active on both
DB files (verified in `_init_audit_db`/`_init_users_db`), and WAL natively
supports concurrent readers plus one writer.

**Tech Stack:** Python stdlib only — `concurrent.futures.ThreadPoolExecutor`,
`sqlite3`, `socket`. No new dependencies.

## Global Constraints

- Implements spec §6 in `docs/specs/2026-08-04-v1.5.0-full-design.md` —
  re-read that section if anything here seems to contradict it.
- Every task must leave the full existing test suite green
  (`cd ~/slurm-deck && python3 -m pytest -q`, currently 894 tests) — this
  is a live production daemon (`slurm-deck-audit.service`); a regression
  here breaks logins/audit for the whole cluster.
- No behavior change visible to callers: a client that triggers a slow
  operation still waits and gets the real result. Only *other* clients
  stop being blocked by it.
- Follow this project's existing style in `deploy/audit_daemon.py`: plain
  functions (no classes), `_log.info`/`_log.warning` for daemon-side
  logging, `(ok, result, error)` tuple returns from handlers (unchanged by
  this plan).
- Deploy/rollout (not part of this plan's tasks, done after merge, same
  as every prior daemon change this project has made): sync
  `/opt/slurm-deck`, `sudo systemctl restart slurm-deck-audit`, verify PID
  actually changed.

---

### Task 1: Self-managed per-request SQLite connections

**Files:**
- Modify: `deploy/audit_daemon.py` (`_handle_connection`, and its call
  site inside `main()`)
- Modify: `tests/test_audit_daemon.py` (`daemon_env` fixture's `_serve()`
  — its hand-rolled accept loop calls `_handle_connection` with the old
  3-argument signature; ~15 existing tests depend on this fixture and
  will fail to even collect correctly if this isn't updated in the same
  commit as the signature change)
- Test: `tests/test_audit_daemon.py` (new test in the same file, next to
  the existing socket-level tests)

**Interfaces:**
- Consumes: `_dispatch(verb, payload, peer_uid, peer_username, audit_conn,
  users_conn)` — **unchanged**, still takes explicit connections.
- Produces: `_handle_connection(conn_sock: socket.socket) -> None` — new
  signature, no longer takes `audit_conn`/`users_conn`. Opens
  `sqlite3.connect(str(DB_PATH))` and `sqlite3.connect(str(USERS_DB))`
  itself, closes both in `finally`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_audit_daemon.py`, near the other socket-level tests
(after `test_socket_peercred_overrides_forged_user`):

```python
def test_handle_connection_opens_its_own_db_connections(tmp_path, monkeypatch):
    """_handle_connection must work when called with ONLY a connected
    socket -- no audit_conn/users_conn arguments -- proving it manages its
    own SQLite connections instead of requiring the caller to supply them."""
    monkeypatch.setenv("AUDIT_STATE", str(tmp_path))
    ad = _load_daemon(tmp_path)
    ad.DB_PATH = tmp_path / "audit.db"
    ad.USERS_DB = tmp_path / "users.db"
    conn = sqlite3.connect(str(ad.DB_PATH))
    ad._init_audit_db(conn)
    conn.close()
    conn = sqlite3.connect(str(ad.USERS_DB))
    ad._init_users_db(conn)
    conn.close()

    server_sock, client_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM)
    req = json.dumps({"verb": "audit.log",
                      "payload": {"action": "test_event"}}).encode()
    client_sock.sendall(struct.pack(">I", len(req)) + req)

    ad._handle_connection(server_sock)

    raw_len = _recv_all(client_sock, 4)
    length = struct.unpack(">I", raw_len)[0]
    resp = json.loads(_recv_all(client_sock, length).decode())
    assert resp["ok"] is True
    client_sock.close()

    check = sqlite3.connect(str(ad.DB_PATH))
    row = check.execute(
        "SELECT action FROM events WHERE action='test_event'").fetchone()
    assert row is not None
    check.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/slurm-deck && python3 -m pytest tests/test_audit_daemon.py::test_handle_connection_opens_its_own_db_connections -v`
Expected: FAIL with `TypeError: _handle_connection() missing 2 required
positional arguments: 'audit_conn' and 'users_conn'`

- [ ] **Step 3: Refactor `_handle_connection`'s signature and body**

In `deploy/audit_daemon.py`, replace:

```python
def _handle_connection(conn_sock: socket.socket,
                       audit_conn: sqlite3.Connection,
                       users_conn: sqlite3.Connection) -> None:
    try:
        conn_sock.settimeout(5.0)
        peer_uid      = _get_peer_uid(conn_sock)
        peer_username = _uid_to_username(peer_uid) if peer_uid is not None else None

        data = _read_message(conn_sock)
        if data is None:
            return

        try:
            req = json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _send_response(conn_sock, False, error=f"bad JSON: {exc}")
            return

        verb    = req.get("verb", "")
        payload = req.get("payload", {})
        ok, result, err = _dispatch(verb, payload, peer_uid,
                                     peer_username, audit_conn, users_conn)
        _send_response(conn_sock, ok, result, err)

    except OSError as exc:
        _log.debug("Connection error: %s", exc)
    finally:
        try:
            conn_sock.close()
        except OSError:
            pass
```

with:

```python
def _handle_connection(conn_sock: socket.socket) -> None:
    """Handles one request end-to-end on whichever thread calls this --
    opens its own short-lived SQLite connections rather than sharing the
    daemon's long-lived ones, so one slow request (mail send, service
    status) never blocks another thread's fast one. Safe without extra
    locking because both DB files already run in WAL mode (see
    _init_audit_db/_init_users_db) -- WAL natively supports concurrent
    readers plus one writer."""
    audit_conn = None
    users_conn = None
    try:
        conn_sock.settimeout(5.0)
        peer_uid      = _get_peer_uid(conn_sock)
        peer_username = _uid_to_username(peer_uid) if peer_uid is not None else None

        data = _read_message(conn_sock)
        if data is None:
            return

        try:
            req = json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _send_response(conn_sock, False, error=f"bad JSON: {exc}")
            return

        verb       = req.get("verb", "")
        payload    = req.get("payload", {})
        audit_conn = sqlite3.connect(str(DB_PATH))
        users_conn = sqlite3.connect(str(USERS_DB))
        ok, result, err = _dispatch(verb, payload, peer_uid,
                                     peer_username, audit_conn, users_conn)
        _send_response(conn_sock, ok, result, err)

    except OSError as exc:
        _log.debug("Connection error: %s", exc)
    finally:
        if audit_conn is not None:
            audit_conn.close()
        if users_conn is not None:
            users_conn.close()
        try:
            conn_sock.close()
        except OSError:
            pass
```

- [ ] **Step 4: Update `main()`'s call site**

In `main()`, find:

```python
                try:
                    conn_sock, _ = server.accept()
                    _handle_connection(conn_sock, audit_conn, users_conn)
                except OSError:
                    pass
```

Replace with:

```python
                try:
                    conn_sock, _ = server.accept()
                    _handle_connection(conn_sock)
                except OSError:
                    pass
```

(The thread pool submission replaces this inline call in Task 2 — this
step only fixes the call site to match the new signature so `main()`
still runs correctly, single-threaded, as an intermediate state.)

- [ ] **Step 5: Update the `daemon_env` fixture's `_serve()` call site**

In `tests/test_audit_daemon.py`, inside the `daemon_env` fixture, find:

```python
    def _serve():
        while running[0]:
            try:
                conn, _ = server.accept()
                ad._handle_connection(conn, audit_conn, users_conn)
            except socket.timeout:
                continue
            except OSError:
                break
        server.close()
```

Replace with:

```python
    def _serve():
        while running[0]:
            try:
                conn, _ = server.accept()
                ad._handle_connection(conn)
            except socket.timeout:
                continue
            except OSError:
                break
        server.close()
```

Leave the fixture's own `ad.DB_PATH`/`ad.USERS_DB` unset at this point —
Step 6 fixes that, since `_handle_connection` now needs those module
attributes to point at the fixture's tmp DB files instead of connecting
via the pre-created `audit_conn`/`users_conn` objects the fixture builds
(which are now unused by the dispatch path, but still returned by the
fixture — several existing tests inspect them directly after making a
request, e.g. querying `users_conn` for a row a request just created).

- [ ] **Step 6: Point the fixture's `DB_PATH`/`USERS_DB` at its tmp files**

Still inside `daemon_env`, immediately after the `ad.JSONL_PATH = ...`
line, add:

```python
    ad.JSONL_PATH = tmp_path / "audit.jsonl"
    ad.DB_PATH = tmp_path / "audit.db"
    ad.USERS_DB = tmp_path / "users.db"
```

This makes the module attribute the fixture's pre-created
`audit_conn`/`users_conn` already point at (`tmp_path / "audit.db"` and
`tmp_path / "users.db"`) the same file `_handle_connection`'s new
self-managed connections will open — so a test that writes via a request
and then reads via the fixture's own `audit_conn`/`users_conn` sees the
same data, just through two separate SQLite connections to the same file
(safe under WAL).

- [ ] **Step 7: Run the new test to verify it passes**

Run: `cd ~/slurm-deck && python3 -m pytest tests/test_audit_daemon.py::test_handle_connection_opens_its_own_db_connections -v`
Expected: PASS

- [ ] **Step 8: Run the full existing test suite**

Run: `cd ~/slurm-deck && python3 -m pytest -q`
Expected: all 894 existing tests plus the 1 new one pass (895 total). Pay
particular attention to every test using the `daemon_env` fixture — this
is exactly the risk Step 5/6 exist to cover.

- [ ] **Step 9: Commit**

```bash
git add deploy/audit_daemon.py tests/test_audit_daemon.py
git commit -m "refactor(audit-daemon): _handle_connection opens its own SQLite connections

Prerequisite for moving connection handling onto a thread pool (next
commit) -- sharing one long-lived audit_conn/users_conn across threads
isn't safe without extra locking, so each request now opens and closes
its own short-lived connections instead. Safe with no new locking code
because both DB files already run PRAGMA journal_mode=WAL.

Updates the daemon_env test fixture's hand-rolled accept loop to match
the new signature, and points its DB_PATH/USERS_DB at the same tmp files
its own pre-created connections use, so existing tests that inspect DB
state after a request still see the same data.

No behavior change for callers -- this is groundwork only, still
single-threaded until the next commit."
```

---

### Task 2: Bounded thread pool + the real concurrency regression test

**Files:**
- Modify: `deploy/audit_daemon.py` (`main()`'s accept loop; new import)
- Test: `tests/test_audit_daemon.py` (new fixture + new test, exercising
  the real `main()` — not the `daemon_env` fixture's hand-rolled loop,
  which deliberately stays single-request-at-a-time since it's testing
  verb *correctness*, not the daemon's concurrency behavior)

**Interfaces:**
- Consumes: `_handle_connection(conn_sock)` from Task 1.
- Produces: no new public interface — `main()`'s external behavior
  (socket path, protocol, responses) is unchanged; only its internal
  dispatch mechanism changes. This is the task that actually fixes the
  bug described in the Goal above.

- [ ] **Step 1: Write a fixture that runs the real `main()`**

Add to `tests/test_audit_daemon.py`:

```python
@pytest.fixture
def real_daemon(tmp_path, monkeypatch):
    """Runs the ACTUAL main() (not a hand-rolled test loop) in a
    background thread, against tmp state -- for tests that need to
    exercise the daemon's real concurrency behavior, not just verb
    correctness. Env vars must be set BEFORE _load_daemon(), since
    SOCKET_PATH/DB_PATH/etc are module-level constants read from
    os.environ at import time."""
    sock_path = str(tmp_path / "test.sock")
    monkeypatch.setenv("AUDIT_SOCKET", sock_path)
    monkeypatch.setenv("AUDIT_STATE", str(tmp_path))
    monkeypatch.setenv("AUDIT_SPOOL", str(tmp_path / "spool"))

    ad = _load_daemon()

    t = threading.Thread(target=ad.main, daemon=True)
    t.start()

    deadline = time.monotonic() + 3.0
    while not Path(sock_path).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("daemon socket never appeared")
        time.sleep(0.05)

    yield ad, sock_path

    ad._running = False
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(sock_path)
    except OSError:
        pass
    t.join(timeout=3)
```

- [ ] **Step 2: Write the failing concurrency test**

```python
def test_slow_mail_send_does_not_block_other_clients(real_daemon, monkeypatch):
    """The actual bug this task fixes: today, one client's slow mail.send
    (a real network call in production) blocks every other client because
    _handle_connection runs inline on main()'s single accept-loop thread.
    This proves a concurrent fast request completes without waiting for
    a slow one."""
    ad, sock_path = real_daemon

    # _h_mail_send requires a non-admin caller to have a registered email
    # (it forces recipient-to-self as an anti-relay measure) -- without
    # this, the request fails immediately with "no registered email for
    # sender" and never reaches _resend_send at all, which would make
    # this test pass for the wrong reason (or not test anything real).
    # Same seeding pattern as test_users_db_email_for_self_allowed.
    real_user = getpass.getuser()
    users_conn = sqlite3.connect(str(ad.USERS_DB))
    ad._h_users_create(
        {"username": real_user, "email": f"{real_user}@test.com", "role": "tool"},
        0, users_conn, _dummy_audit_conn(ad))
    users_conn.close()

    def _slow_resend_send(*args, **kwargs):
        time.sleep(2.0)
        return True, {"id": "fake"}, ""

    monkeypatch.setattr(ad, "_resend_send", _slow_resend_send)

    results = {}

    def _slow_client():
        start = time.monotonic()
        _send_req(sock_path, "mail.send",
                  {"to": "alice@example.com", "subject": "s", "html": "h"})
        results["slow_elapsed"] = time.monotonic() - start

    def _fast_client():
        time.sleep(0.2)  # let the slow request start first
        start = time.monotonic()
        _send_req(sock_path, "audit.log", {"action": "fast_event"})
        results["fast_elapsed"] = time.monotonic() - start

    t_slow = threading.Thread(target=_slow_client)
    t_fast = threading.Thread(target=_fast_client)
    t_slow.start()
    t_fast.start()
    t_slow.join(timeout=5)
    t_fast.join(timeout=5)

    assert "fast_elapsed" in results, "fast client never got a response"
    assert results["fast_elapsed"] < 1.0, (
        f"fast request took {results['fast_elapsed']:.2f}s -- "
        "it was blocked behind the slow one instead of running concurrently"
    )
```

Note: `_h_mail_send` (the verb handler `mail.send` dispatches to) calls
`_resend_send` internally — `monkeypatch.setattr(ad, "_resend_send", ...)`
patches the module-level function, which the handler looks up by name at
call time, so this correctly intercepts it without needing to know
`_h_mail_send`'s internals beyond that it's the thing that ultimately
calls `_resend_send`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd ~/slurm-deck && python3 -m pytest tests/test_audit_daemon.py::test_slow_mail_send_does_not_block_other_clients -v`
Expected: FAIL — `fast_elapsed` will be close to 2 seconds (or the test
will time out waiting for the fast client), because `main()` still
handles connections inline, one at a time, after Task 1 alone.

- [ ] **Step 4: Add the thread pool to `main()`**

In `deploy/audit_daemon.py`, add to the imports near the top:

```python
from concurrent.futures import ThreadPoolExecutor
```

In `main()`, find:

```python
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    sock_path.chmod(0o777)   # world-connectable; SO_PEERCRED is the security boundary
    server.listen(16)
    server.setblocking(False)

    _log.info("Listening on %s (STREAM, SO_PEERCRED)", SOCKET_PATH)
    _drain_spool(audit_conn)

    last_drain = time.monotonic()
    while _running:
        readable, _, _ = select.select([server], [], [], 5.0)
        for s in readable:
            if s is server:
                try:
                    conn_sock, _ = server.accept()
                    _handle_connection(conn_sock)
                except OSError:
                    pass
        if time.monotonic() - last_drain > 30:
            _drain_spool(audit_conn)
            last_drain = time.monotonic()

    server.close()
```

Replace with:

```python
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    sock_path.chmod(0o777)   # world-connectable; SO_PEERCRED is the security boundary
    server.listen(16)
    server.setblocking(False)

    _log.info("Listening on %s (STREAM, SO_PEERCRED)", SOCKET_PATH)
    _drain_spool(audit_conn)

    # Bounded so a burst of connections can't spawn unbounded threads.
    # _handle_connection opens its own short-lived SQLite connections
    # (Task 1), so a slow one (mail.send, service.status) no longer
    # blocks this accept loop or any other worker thread.
    executor = ThreadPoolExecutor(max_workers=8)

    last_drain = time.monotonic()
    while _running:
        readable, _, _ = select.select([server], [], [], 5.0)
        for s in readable:
            if s is server:
                try:
                    conn_sock, _ = server.accept()
                    executor.submit(_handle_connection, conn_sock)
                except OSError:
                    pass
        if time.monotonic() - last_drain > 30:
            _drain_spool(audit_conn)
            last_drain = time.monotonic()

    executor.shutdown(wait=True)
    server.close()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd ~/slurm-deck && python3 -m pytest tests/test_audit_daemon.py::test_slow_mail_send_does_not_block_other_clients -v`
Expected: PASS — `fast_elapsed` well under 1 second, proving it wasn't
serialized behind the 2-second slow request.

- [ ] **Step 6: Run the full existing test suite**

Run: `cd ~/slurm-deck && python3 -m pytest -q`
Expected: all tests pass (896 total including both new tests from Tasks 1
and 2).

- [ ] **Step 7: Commit**

```bash
git add deploy/audit_daemon.py tests/test_audit_daemon.py
git commit -m "fix(audit-daemon): bounded thread pool so one slow request can't block every client

main()'s accept loop called _handle_connection inline on a single thread
-- a slow mail.send (real Resend API call, up to 20s) or service.status
check (systemctl+journalctl, up to 15s) froze every other user's login,
job submission, file access, and audit logging for that whole duration.

Connections now go through a bounded ThreadPoolExecutor(max_workers=8)
instead. Callers see no behavior change -- the client that triggered the
slow operation still waits for its own real result; only other clients
stop being blocked by it. Safe with Task 1's per-request SQLite
connections and the DB files' existing WAL mode, no new locking code.

New regression test proves the actual bug: a concurrent fast request no
longer waits behind a mocked 2-second-slow mail.send.

896 tests pass."
```

---

### Task 3: Graceful shutdown drains in-flight requests

**Files:**
- Test: `tests/test_audit_daemon.py` (new test)

**Interfaces:**
- Consumes: `executor.shutdown(wait=True)` from Task 2's `main()` change
  (already written — this task only verifies it's correct, since
  `wait=True` was already used in Task 2's Step 4).
- Produces: nothing new — this task is verification, not new
  functionality. Included as its own task because "does shutdown actually
  wait for in-flight work" is exactly the kind of thing that's easy to
  get subtly wrong (e.g. `wait=False`) and deserves its own explicit,
  independently-reviewable proof rather than being an assumed side effect
  of Task 2.

- [ ] **Step 1: Write the test**

```python
def test_shutdown_waits_for_in_flight_request(real_daemon, monkeypatch):
    """_running=False must not drop a request that's already in flight --
    executor.shutdown(wait=True) in main() is what guarantees this."""
    ad, sock_path = real_daemon

    # Same seeding requirement as Task 2's test -- see its comment for why.
    real_user = getpass.getuser()
    users_conn = sqlite3.connect(str(ad.USERS_DB))
    ad._h_users_create(
        {"username": real_user, "email": f"{real_user}@test.com", "role": "tool"},
        0, users_conn, _dummy_audit_conn(ad))
    users_conn.close()

    def _slow_resend_send(*args, **kwargs):
        time.sleep(1.0)
        return True, {"id": "fake"}, ""

    monkeypatch.setattr(ad, "_resend_send", _slow_resend_send)

    result = {}

    def _slow_client():
        result["resp"] = _send_req(sock_path, "mail.send",
                                   {"to": "alice@example.com",
                                    "subject": "s", "html": "h"})

    t = threading.Thread(target=_slow_client)
    t.start()
    time.sleep(0.2)  # let the slow request actually start

    ad._running = False  # signal shutdown while the request is in flight

    t.join(timeout=5)
    assert "resp" in result
    assert result["resp"]["ok"] is True, (
        "in-flight request was dropped or errored during shutdown "
        "instead of being allowed to complete"
    )
```

- [ ] **Step 2: Run the test**

Run: `cd ~/slurm-deck && python3 -m pytest tests/test_audit_daemon.py::test_shutdown_waits_for_in_flight_request -v`
Expected: PASS immediately — Task 2 already wrote `executor.shutdown(wait=True)`,
so this step is confirming that decision was correct, not implementing
something new. If it fails, the bug is that shutdown doesn't actually
wait — check `executor.shutdown(wait=True)` is really in place (not
`wait=False`, and not skipped by an early `return`/exception path in
`main()`).

- [ ] **Step 3: Run the full existing test suite**

Run: `cd ~/slurm-deck && python3 -m pytest -q`
Expected: all tests pass (897 total).

- [ ] **Step 4: Commit**

```bash
git add tests/test_audit_daemon.py
git commit -m "test(audit-daemon): verify graceful shutdown drains in-flight requests

Explicit regression test for executor.shutdown(wait=True) in main()
(added as part of the Task 2 thread-pool change) -- a request already in
flight when _running goes False must complete and respond, not be
dropped. Verification only, no functional change.

897 tests pass."
```

---

## Self-review notes (for whoever executes this plan)

- **Spec coverage**: this plan implements spec §6 in full — the
  thread-pool design, per-thread connections, and the "callers see no
  behavior change" property are all directly tested (Task 2's test proves
  concurrency; Task 3's proves no in-flight work is dropped on shutdown).
- **What this plan does NOT cover**: §7 (disk quotas) and §8 (bulk
  provisioning) are separate plans, per the spec's own phasing in §10 —
  don't fold them into this work even if the diff is small, they're
  independently reviewable pieces.
- **Watch for**: Task 1's Step 5/6 fixture update is the highest-risk step
  in this whole plan — it's easy to update the signature but miss one of
  the ~15 existing tests that depends on `daemon_env` behaving exactly as
  before. Step 8 (full suite run) is not optional.
