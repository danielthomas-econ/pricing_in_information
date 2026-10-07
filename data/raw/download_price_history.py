#!/usr/bin/env python3
"""
Polymarket CLOB price-history bulk downloader
===============================================

Builds a resumable panel dataset of price history for a large list of
Polymarket binary (Yes/No) markets, using the CLOB API's
GET /prices-history and POST /batch-prices-history endpoints.

WHY THIS ISN'T A TRIVIAL requests.get() LOOP
----------------------------------------------
1. Bug/quirk in /prices-history: calling it with interval="max" plus a
   fine fidelity (below ~12h = 720 min) silently returns an EMPTY history
   for markets that have already resolved/closed, even for markets that
   had heavy trading. This is documented in the wild (see
   Polymarket/py-clob-client issues #189 and #216) and is NOT mentioned
   anywhere in the official docs. The confirmed community workaround is
   to never rely on `interval` and always pass explicit startTs/endTs
   windows, chunked into ranges of a couple weeks. This script always
   does that.

2. /prices-history takes exactly one token (asset) id per call. There is
   also POST /batch-prices-history, which accepts up to 20 token ids per
   call -- but all 20 share one start_ts/end_ts/fidelity. To exploit this
   at scale (262k markets), this script aligns every market's history
   request onto a shared, fixed calendar grid of chunks (e.g. 15-day
   windows anchored at a fixed epoch) instead of per-market-relative
   windows. That means many markets active in the same real-world period
   land in the same chunk and can be fetched in ONE batched HTTP call for
   up to 20 of them, instead of 20 separate calls. At your scale this is
   the difference between an afternoon and a week, given the API's
   ~100 req/s budget.

3. CLOB read endpoints (including /prices-history) are NOT subject to
   Polymarket's geoblock -- that only blocks order placement. No VPN
   needed to run this.

4. Resumability: with 262.5k markets this WILL get interrupted at some
   point (network blip, laptop sleep, rate-limit backoff spiral, etc).
   Every (token_id, chunk) pair that has been successfully fetched (even
   if the result was legitimately empty) is recorded in a local SQLite
   DB before any in-memory result is discarded, so re-running the script
   just picks up where it left off. All raw (t, p) rows are also written
   to that same DB as you go, not held in memory.

INPUT
-----
A markets file (parquet or csv) with (at minimum) these columns, matching
Gamma API's own field names so you can pass your already-filtered Gamma
export straight in:
    - id  OR  conditionId          (a unique market identifier)
    - clobTokenIds                 (JSON string of a 2-element list, e.g.
                                     '["1234...", "5678..."]')
    - outcomes                     (JSON string list, e.g. '["Yes","No"]').
                                     If missing, index 0 is assumed to be
                                     "Yes".
    - startDate  OR  createdAt     (market open timestamp, ISO string or
                                     unix seconds)
    - endDate  OR  closedTime      (market resolution timestamp)

If your column names differ, edit COLUMN_ALIASES below rather than
renaming your file.

OUTPUT
------
A SQLite database (default ./price_history.db) with:
    - price_history(token_id, market_id, t, p, fidelity_min)
    - done_chunks(token_id, chunk_start, chunk_end)      -- resume ledger
    - done_markets(token_id)                             -- fully-fetched markers
    - failed_chunks(token_id, chunk_start, chunk_end, error)

Export to Parquet/CSV afterwards with export_to_parquet.py (included).

USAGE
-----
    pip install aiohttp pandas pyarrow --break-system-packages

    python download_price_history.py \
        --input filtered_markets.parquet \
        --db price_history.db \
        --fidelity 1 \
        --chunk-days 15 \
        --concurrency 8 \
        --rate-limit 80

Flags are explained with --help.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Config you may need to tweak for your Gamma export's exact column names
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "id": ["id", "conditionId", "condition_id", "market_id"],
    "clob_token_ids": ["clobTokenIds", "clob_token_ids"],
    "outcomes": ["outcomes"],
    "start": ["startDate", "createdAt", "start_date", "created_at"],
    "end": ["closedTime", "endDate", "closed_time", "end_date"],
}

CLOB_BASE = "https://clob.polymarket.com"
PRICES_HISTORY_URL = f"{CLOB_BASE}/prices-history"
BATCH_PRICES_HISTORY_URL = f"{CLOB_BASE}/batch-prices-history"
BATCH_MAX_MARKETS = 20  # hard API limit

# Fixed epoch anchor so chunk boundaries are identical across ALL markets,
# which is what lets us batch different markets that overlap in real time.
CHUNK_ANCHOR_TS = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polymarket-dl")


# ---------------------------------------------------------------------------
# SQLite sink (single writer thread/coroutine, everyone else just enqueues)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS price_history (
    token_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    t INTEGER NOT NULL,
    p REAL NOT NULL,
    fidelity_min INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS done_chunks (
    token_id TEXT NOT NULL,
    chunk_start INTEGER NOT NULL,
    chunk_end INTEGER NOT NULL,
    PRIMARY KEY (token_id, chunk_start, chunk_end)
);
CREATE TABLE IF NOT EXISTS done_markets (
    token_id TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS failed_chunks (
    token_id TEXT NOT NULL,
    chunk_start INTEGER NOT NULL,
    chunk_end INTEGER NOT NULL,
    error TEXT,
    PRIMARY KEY (token_id, chunk_start, chunk_end)
);
CREATE INDEX IF NOT EXISTS idx_ph_token ON price_history(token_id);
"""


class Store:
    def __init__(self, path: str, restore_from: Optional[str] = None):
        # If the live (local-disk) DB doesn't exist yet but a prior backup
        # does, restore it first -- this is what makes "local disk for
        # safety, Drive for durability" transparent across a Colab restart.
        if restore_from and not Path(path).exists() and Path(restore_from).exists():
            log.info(f"No local DB at {path}; restoring from backup {restore_from}...")
            shutil.copy2(restore_from, path)
            log.info("Restore complete.")

        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def already_done_chunks(self, token_ids: list[str]) -> set[tuple[str, int, int]]:
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT token_id, chunk_start, chunk_end FROM done_chunks "
            f"WHERE token_id IN ({','.join('?' * len(token_ids))})",
            token_ids,
        ) if token_ids else None
        return set(cur.fetchall()) if token_ids else set()

    def already_done_markets(self) -> set[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT token_id FROM done_markets")
        return {r[0] for r in cur.fetchall()}

    def write_batch(self, rows, done_chunk_keys, failed_chunk_keys):
        cur = self.conn.cursor()
        if rows:
            cur.executemany(
                "INSERT INTO price_history (token_id, market_id, t, p, fidelity_min) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        if done_chunk_keys:
            cur.executemany(
                "INSERT OR IGNORE INTO done_chunks (token_id, chunk_start, chunk_end) "
                "VALUES (?, ?, ?)",
                done_chunk_keys,
            )
        if failed_chunk_keys:
            cur.executemany(
                "INSERT OR REPLACE INTO failed_chunks "
                "(token_id, chunk_start, chunk_end, error) VALUES (?, ?, ?, ?)",
                failed_chunk_keys,
            )
        self.conn.commit()

    def mark_market_done(self, token_id: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO done_markets (token_id) VALUES (?)", (token_id,)
        )
        self.conn.commit()

    def stats(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM price_history")
        rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM done_markets")
        done = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM failed_chunks")
        failed = cur.fetchone()[0]
        return rows, done, failed


def safe_backup(src_path: str, dst_path: str):
    """Atomic, consistency-safe snapshot of a LIVE (possibly concurrently
    written) SQLite DB, using SQLite's own backup API rather than a raw file
    copy. A raw copy of a database that's actively being written to can
    itself capture a torn/inconsistent state; sqlite3's backup() reads
    through SQLite's own page-level API and is safe to run concurrently
    with writers. Written to a .tmp path first and atomically renamed only
    once complete, so a crash mid-backup never leaves a half-written file
    at the real destination.

    IMPORTANT: dst_path should NOT be the live path other processes are
    writing to (e.g. don't back up directly onto a Drive mount if Drive
    itself is flaky -- back up TO local disk, or treat the Drive copy as
    "best effort" only).
    """
    tmp_path = dst_path + ".tmp"
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(tmp_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    os.replace(tmp_path, dst_path)


# ---------------------------------------------------------------------------
# Market loading + parsing
# ---------------------------------------------------------------------------
@dataclass
class Market:
    market_id: str
    yes_token: str
    no_token: Optional[str]
    start_ts: int
    end_ts: int


def _pick_col(df_columns, candidates):
    for c in candidates:
        if c in df_columns:
            return c
    return None


def _to_unix(val) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = int(val)
        # heuristic: ms vs s
        return v // 1000 if v > 10_000_000_000 else v
    s = str(val)
    try:
        return int(float(s))
    except ValueError:
        pass
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def load_markets(path: str, both_outcomes: bool) -> list[Market]:
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    cols = set(df.columns)
    id_col = _pick_col(cols, COLUMN_ALIASES["id"])
    tok_col = _pick_col(cols, COLUMN_ALIASES["clob_token_ids"])
    out_col = _pick_col(cols, COLUMN_ALIASES["outcomes"])
    start_col = _pick_col(cols, COLUMN_ALIASES["start"])
    end_col = _pick_col(cols, COLUMN_ALIASES["end"])

    missing = [
        name
        for name, col in [
            ("id", id_col),
            ("clobTokenIds", tok_col),
            ("start", start_col),
            ("end", end_col),
        ]
        if col is None
    ]
    if missing:
        raise SystemExit(
            f"Could not find columns for: {missing}. "
            f"Available columns: {sorted(cols)}. "
            f"Edit COLUMN_ALIASES at the top of this script if your export "
            f"uses different names."
        )

    markets = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            tokens = json.loads(row[tok_col])
            if not isinstance(tokens, list) or len(tokens) < 1:
                skipped += 1
                continue
            yes_idx = 0
            if out_col is not None and row.get(out_col):
                try:
                    outcomes = json.loads(row[out_col])
                    for i, o in enumerate(outcomes):
                        if str(o).strip().lower() == "yes":
                            yes_idx = i
                            break
                except Exception:
                    pass
            yes_token = str(tokens[yes_idx])
            no_token = None
            if both_outcomes and len(tokens) > 1:
                other_idx = 1 - yes_idx if len(tokens) == 2 else (yes_idx + 1) % len(tokens)
                no_token = str(tokens[other_idx])

            start_ts = _to_unix(row[start_col])
            end_ts = _to_unix(row[end_col])
            if start_ts is None or end_ts is None or end_ts <= start_ts:
                skipped += 1
                continue

            markets.append(
                Market(
                    market_id=str(row[id_col]),
                    yes_token=yes_token,
                    no_token=no_token,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )
        except Exception:
            skipped += 1
            continue

    if skipped:
        log.warning(f"Skipped {skipped} rows that failed to parse.")
    log.info(f"Loaded {len(markets)} markets.")
    return markets


# ---------------------------------------------------------------------------
# Chunk grid
# ---------------------------------------------------------------------------
def chunks_for(start_ts: int, end_ts: int, chunk_secs: int):
    """Yield (chunk_start, chunk_end) on a FIXED global grid so different
    markets active in the same real period reuse identical boundaries,
    which is required for batching them together."""
    first_idx = (start_ts - CHUNK_ANCHOR_TS) // chunk_secs
    last_idx = (end_ts - CHUNK_ANCHOR_TS) // chunk_secs
    for idx in range(first_idx, last_idx + 1):
        cs = CHUNK_ANCHOR_TS + idx * chunk_secs
        ce = cs + chunk_secs
        yield max(cs, start_ts), min(ce, end_ts)


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


# ---------------------------------------------------------------------------
# HTTP fetch with retry/backoff
# ---------------------------------------------------------------------------
async def fetch_json(session, limiter, method, url, *, params=None, json_body=None, retries=4):
    for attempt in range(retries):
        await limiter.acquire()
        try:
            async with session.request(
                method, url, params=params, json=json_body, timeout=aiohttp.ClientTimeout(total=12)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    body = {}
                    try:
                        body = await resp.json()
                    except Exception:
                        pass
                    wait = min(10, body.get("retry_after_seconds") or (2 ** attempt))
                    log.warning(f"429 rate-limited, waiting {wait}s")
                    await asyncio.sleep(float(wait) + random.random())
                    continue
                if 500 <= resp.status < 600:
                    wait = min(8, 2 ** attempt) + random.random()
                    log.warning(f"{resp.status} server error on {url}, retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = min(8, 2 ** attempt) + random.random()
            log.warning(f"Network error ({e!r}), retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Exceeded retries for {url} params={params} body={json_body}")


FALLBACK_FIDELITY_MIN = 720  # 12h - known to work for closed markets when finer fails


async def fetch_single_fallback(session, limiter, token_id, cs, ce, fidelity):
    """Individual GET call, used for (a) leftover markets that don't share a
    batch, and (b) re-checking a chunk that came back empty from the batch
    call, at coarser fidelity, to tell 'API quirk' apart from 'genuinely no
    trades in this window'."""
    params = {"market": token_id, "startTs": cs, "endTs": ce, "fidelity": fidelity}
    data = await fetch_json(session, limiter, "GET", PRICES_HISTORY_URL, params=params)
    hist = data.get("history", [])
    if not hist and fidelity < FALLBACK_FIDELITY_MIN:
        params["fidelity"] = FALLBACK_FIDELITY_MIN
        data = await fetch_json(session, limiter, "GET", PRICES_HISTORY_URL, params=params)
        hist = data.get("history", [])
        fidelity = FALLBACK_FIDELITY_MIN if hist else fidelity
    return hist, fidelity


async def fetch_batch(session, limiter, token_ids, cs, ce, fidelity):
    body = {
        "markets": token_ids,
        "start_ts": cs,
        "end_ts": ce,
        "fidelity": fidelity,
    }
    data = await fetch_json(session, limiter, "POST", BATCH_PRICES_HISTORY_URL, json_body=body)
    return data.get("history", {})


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def run(args):
    store = Store(args.db, restore_from=args.backup_path)
    both = args.both_outcomes
    markets = load_markets(args.input, both)

    already_done_markets = store.already_done_markets()
    markets = [m for m in markets if m.yes_token not in already_done_markets]
    log.info(f"{len(markets)} markets remaining after resume-skip.")

    chunk_secs = args.chunk_days * 86400
    fidelity = args.fidelity

    # token_id -> market (for bookkeeping / writing market_id into rows)
    token_to_market: dict[str, str] = {}
    # chunk_key -> list of token_ids needing it
    chunk_map: dict[tuple[int, int], list[str]] = {}
    token_all_chunks: dict[str, list[tuple[int, int]]] = {}

    for idx, m in enumerate(markets):
        tokens = [m.yes_token] + ([m.no_token] if both and m.no_token else [])
        needed_chunks = list(chunks_for(m.start_ts, m.end_ts, chunk_secs))
        for tok in tokens:
            token_to_market[tok] = m.market_id
            token_all_chunks[tok] = needed_chunks
            for ck in needed_chunks:
                chunk_map.setdefault(ck, []).append(tok)
        if (idx + 1) % 20000 == 0:
            log.info(f"Built chunk grid for {idx + 1}/{len(markets)} markets...")

    # drop already-done (token, chunk) pairs
    # (fetch the done-set ONCE across all batches, then filter chunk_map ONCE --
    # the previous version re-scanned all of chunk_map inside the batch loop,
    # which made this step scale as O(n_tokens * n_chunks) instead of
    # O(n_tokens + n_chunks). On a fresh/empty DB this whole block is nearly
    # instant since done_pairs is empty; it only matters on resumed runs.)
    all_tokens = list(token_to_market.keys())
    log.info(f"Checking resume state for {len(all_tokens)} tokens...")
    done_pairs: set[tuple[str, int, int]] = set()
    for i in range(0, len(all_tokens), 500):
        batch = all_tokens[i : i + 500]
        done_pairs |= store.already_done_chunks(batch)
    if done_pairs:
        for ck in chunk_map:
            chunk_map[ck] = [t for t in chunk_map[ck] if (t, ck[0], ck[1]) not in done_pairs]

    chunk_map = {k: v for k, v in chunk_map.items() if v}
    total_chunk_calls = sum(
        (len(v) + BATCH_MAX_MARKETS - 1) // BATCH_MAX_MARKETS for v in chunk_map.values()
    )
    log.info(
        f"{len(chunk_map)} distinct time windows to fetch, "
        f"~{total_chunk_calls} HTTP calls after batching."
    )

    limiter = RateLimiter(args.rate_limit)
    remaining_chunks: dict[str, int] = {t: len(cks) for t, cks in token_all_chunks.items()}

    queue: asyncio.Queue = asyncio.Queue()
    for ck, toks in chunk_map.items():
        queue.put_nowait((ck, toks))
    total_jobs = queue.qsize()

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    progress = {"completed": 0}

    async with aiohttp.ClientSession(connector=connector) as session:

        async def resolve_token(tok, cs, ce, prefetched_hist):
            """Resolve one token's history for this chunk, given whatever the
            batch call already returned for it (possibly empty). Isolated in
            its own try/except so ONE token failing all its retries doesn't
            discard results already fetched for the other ~19 tokens sharing
            this batch call."""
            hist = prefetched_hist
            used_fidelity = fidelity
            try:
                if not hist:
                    hist, used_fidelity = await fetch_single_fallback(
                        session, limiter, tok, cs, ce, fidelity
                    )
                return tok, hist, used_fidelity, None
            except Exception as e:
                return tok, None, None, e

        async def process_one(ck, tokens):
            cs, ce = ck
            rows = []
            done_keys = []
            failed_keys = []

            if len(tokens) > 1:
                for i in range(0, len(tokens), BATCH_MAX_MARKETS):
                    group = tokens[i : i + BATCH_MAX_MARKETS]
                    try:
                        result = await fetch_batch(session, limiter, group, cs, ce, fidelity)
                    except Exception as e:
                        # Whole batch call failed -- don't give up on every
                        # token in it; let each fall through to its own
                        # individual (concurrent) fallback attempt instead.
                        log.warning(f"Batch call failed for chunk {ck} ({e!r}), falling back per-token")
                        result = {}
                    token_results = await asyncio.gather(
                        *[resolve_token(tok, cs, ce, result.get(tok, [])) for tok in group]
                    )
                    for tok, hist, used_fidelity, err in token_results:
                        if err is not None:
                            failed_keys.append((tok, cs, ce, str(err)))
                            continue
                        for pt in hist:
                            rows.append(
                                (tok, token_to_market[tok], int(pt["t"]), float(pt["p"]), used_fidelity)
                            )
                        done_keys.append((tok, cs, ce))
            else:
                tok, hist, used_fidelity, err = await resolve_token(tokens[0], cs, ce, [])
                if err is not None:
                    failed_keys.append((tok, cs, ce, str(err)))
                else:
                    for pt in hist:
                        rows.append(
                            (tok, token_to_market[tok], int(pt["t"]), float(pt["p"]), used_fidelity)
                        )
                    done_keys.append((tok, cs, ce))

            store.write_batch(rows, done_keys, failed_keys)
            for tok, s, e in done_keys:
                remaining_chunks[tok] -= 1
                if remaining_chunks[tok] <= 0:
                    store.mark_market_done(tok)

        async def worker(worker_id):
            while True:
                try:
                    ck, toks = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await process_one(ck, toks)
                finally:
                    progress["completed"] += 1
                    progress["last_activity"] = time.monotonic()
                    if progress["completed"] % 200 == 0:
                        rows, done_m, failed = store.stats()
                        elapsed = time.monotonic() - start_time
                        log.info(
                            f"[{progress['completed']}/{total_jobs} chunk-tasks] "
                            f"{rows:,} rows | {done_m:,} markets fully done | "
                            f"{failed:,} failed chunks | {elapsed/60:.1f} min elapsed"
                        )
                    queue.task_done()

        async def watchdog():
            # If NOTHING completes (success or failure) for --stall-timeout
            # seconds, the network is stuck in a state our per-request
            # retry/backoff isn't escaping on its own (observed in practice:
            # sustained outages can leave workers backing off for 5-10+
            # minutes). Rather than wait that out, force-exit immediately.
            # This skips graceful cleanup on purpose -- a hung aiohttp
            # connector is exactly what we're trying to get away from, and
            # every completed chunk is already committed to SQLite, so a
            # hard exit here loses nothing. Pair this with an outer loop
            # (see the shell wrapper) that relaunches the script; resuming
            # picks up from done_chunks/done_markets automatically.
            while True:
                await asyncio.sleep(15)
                idle = time.monotonic() - progress["last_activity"]
                if idle > args.stall_timeout:
                    log.error(
                        f"No progress for {idle:.0f}s (> --stall-timeout "
                        f"{args.stall_timeout}s) -- forcing hard restart."
                    )
                    sys.stderr.flush()
                    os._exit(1)

        async def backup_loop():
            if not args.backup_path:
                return
            while True:
                await asyncio.sleep(args.backup_interval)
                try:
                    await asyncio.to_thread(safe_backup, args.db, args.backup_path)
                    log.info(f"Backed up DB to {args.backup_path}")
                except Exception as e:
                    log.warning(f"Backup failed ({e!r}), will retry next interval")

        log.info(f"Launching {args.concurrency} workers over {total_jobs} chunk-jobs...")
        start_time = time.monotonic()
        progress["last_activity"] = start_time
        watchdog_task = asyncio.create_task(watchdog())
        backup_task = asyncio.create_task(backup_loop())
        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        # gather with return_exceptions so a KeyboardInterrupt/CancelledError on
        # one worker doesn't hide the others' state; real cancellation on
        # Ctrl-C is handled by asyncio.run()'s own shutdown in main(), and since
        # there are only --concurrency workers (not one task per chunk), that
        # shutdown is now fast regardless of how many chunk-jobs remain.
        await asyncio.gather(*workers, return_exceptions=True)
        watchdog_task.cancel()
        backup_task.cancel()

    if args.backup_path:
        try:
            safe_backup(args.db, args.backup_path)
            log.info(f"Final backup written to {args.backup_path}")
        except Exception as e:
            log.warning(f"Final backup failed: {e!r}")

    rows, done_m, failed = store.stats()
    log.info(f"DONE. {rows:,} rows, {done_m:,} markets complete, {failed:,} chunks failed.")
    if failed:
        log.info("Re-run this script to retry failed chunks (they are not marked done).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to filtered markets file (.parquet or .csv)")
    ap.add_argument(
        "--db",
        default="price_history.db",
        help="SQLite output path. IMPORTANT: keep this on LOCAL disk (e.g. "
        "/content/price_history.db in Colab), never on a Drive/network "
        "mount -- SQLite (especially WAL mode) is not safe on FUSE-mounted "
        "filesystems and this can corrupt the database. Use --backup-path "
        "to get durability on Drive instead.",
    )
    ap.add_argument(
        "--backup-path",
        default=None,
        help="Optional path (can be on Drive) that gets a safe, atomic "
        "snapshot of --db every --backup-interval seconds via SQLite's "
        "own backup API. If --db doesn't exist yet at startup but this "
        "path does, it's restored from here automatically -- this is what "
        "lets progress survive a Colab disconnect/restart without ever "
        "writing the live DB to Drive directly.",
    )
    ap.add_argument(
        "--backup-interval",
        type=float,
        default=1800,
        help="Seconds between backups to --backup-path (default 1800 = 30 "
        "min). WARNING: on Google Drive, overwriting an existing file "
        "repeatedly often gets treated as 'delete old version, create "
        "new' under the hood, and Drive's Trash counts against your "
        "quota until emptied. At short intervals over a multi-hour run "
        "this can silently consume many times the current file's size in "
        "Trash. Keep this interval long (30-60+ min) for large DBs, and "
        "periodically empty Drive's Trash from drive.google.com.",
    )
    ap.add_argument("--fidelity", type=int, default=60, help="Minutes per price point (default 60 = hourly)")
    ap.add_argument("--chunk-days", type=int, default=15, help="Calendar chunk size in days (default 15)")
    ap.add_argument("--concurrency", type=int, default=8, help="Concurrent in-flight requests")
    ap.add_argument("--rate-limit", type=float, default=80, help="Max requests/sec (CLOB cap is ~100/s)")
    ap.add_argument("--both-outcomes", action="store_true", help="Fetch No token too, not just Yes")
    ap.add_argument(
        "--stall-timeout",
        type=float,
        default=90,
        help="Seconds with zero progress before force-exiting (default 90). "
        "Pair with an outer restart loop so it auto-resumes.",
    )
    args = ap.parse_args()

    # NOTE: deliberately NOT installing a custom SIGINT handler that touches
    # logging/stdout. A signal handler can fire while another part of the
    # program is mid-write to the same stream, and re-entering logging (or
    # print) from inside it can crash with "reentrant call inside
    # <_io.BufferedWriter>". Instead we just let Ctrl-C raise the normal
    # KeyboardInterrupt and catch it here, in ordinary (non-signal-handler)
    # code, where logging is safe.
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.warning("Interrupted -- progress is saved in the DB, just re-run to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
