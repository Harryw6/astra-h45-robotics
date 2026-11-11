"""Metadata retrieval with SQLite and exact q6 neighbors with scikit-learn.

This index makes no semantic, task-stage, grasp-success or quality judgement.
Original episode and block metadata are retained, including absent arm state.
"""
from __future__ import annotations

import hashlib
from contextlib import closing
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np


INDEX_FILENAME = "retrieval.sqlite3"


def _q6(values):
    try:
        vector = np.asarray(values, dtype=float)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("joint_positions must be finite q6") from None
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("joint_positions must be finite q6")
    return vector


def _record_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("record_path must be a relative path within library_root")
    absolute = (root / relative).resolve()
    if not absolute.is_relative_to(root):
        raise ValueError("record_path must remain within library_root")
    return absolute


def _nonnegative_seconds(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite nonnegative seconds") from None
    if not math.isfinite(result) or result < 0 or isinstance(value, bool):
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return result


def build_index(library_root) -> dict:
    """Build/atomically replace retrieval.sqlite3 from manifest and blocks.

    No source record, episode or quality flag is removed. Missing or invalid
    q_start is recorded as an exclusion reason for joint similarity only.
    Vectors are JSON in SQLite; no pickled fitted estimator is persisted.
    """
    root = Path(library_root).resolve()
    manifest_path, blocks_path = root / "manifest.json", root / "blocks.jsonl"
    manifest_bytes, blocks_bytes = manifest_path.read_bytes(), blocks_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    episodes = manifest["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("manifest episodes must be a list")
    episode_rows, episode_ids = [], set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
            raise ValueError("episode_id must be unique and nonempty")
        episode_ids.add(episode_id)
        episode_rows.append((episode_id, episode.get("date"), json.dumps(episode, ensure_ascii=False)))
    block_rows, event_rows, block_ids = [], [], set()
    eligible_count = 0
    for line_number, line in enumerate(blocks_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        block = json.loads(line)
        block_id = block.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(f"invalid block_id at line {line_number}")
        if block_id in block_ids:
            raise ValueError(f"duplicate block_id: {block_id}")
        block_ids.add(block_id)
        if block.get("episode_id") not in episode_ids:
            raise ValueError(f"block episode_id missing from manifest: {block_id}")
        if block.get("kind") not in {"fixed", "dense"}:
            raise ValueError(f"invalid block kind: {block_id}")
        start_s = _nonnegative_seconds(block["start_s"], "start_s")
        end_s = _nonnegative_seconds(block["end_s"], "end_s")
        if end_s < start_s:
            raise ValueError(f"block end_s precedes start_s: {block_id}")
        _record_path(root, block["record_path"])
        events = block.get("events", [])
        if not isinstance(events, list) or not all(isinstance(e, str) for e in events):
            raise ValueError(f"events must be strings: {block_id}")
        exclusion_reason, vector = None, None
        if block.get("q_start") is None:
            exclusion_reason = "missing_joint_positions"
        else:
            try:
                vector = _q6(block["q_start"]).tolist()
            except ValueError:
                exclusion_reason = "joint_positions_must_be_finite_q6"
        eligible = vector is not None
        eligible_count += int(eligible)
        block_rows.append((
            block_id, block["episode_id"], block.get("date"), start_s, end_s,
            block["kind"], block["record_path"], int(eligible), exclusion_reason,
            json.dumps(vector, allow_nan=False), json.dumps(block, ensure_ascii=False),
        ))
        event_rows.extend((block_id, event) for event in sorted(set(events)))
    if manifest.get("block_count", len(block_rows)) != len(block_rows):
        raise ValueError("manifest block_count does not match blocks.jsonl")

    # Write into a sibling temporary database so an invalid rebuild preserves
    # the previous usable snapshot. Only this generated index is replaced.
    descriptor, temp_name = tempfile.mkstemp(prefix="retrieval-", suffix=".sqlite3.tmp", dir=root)
    os.close(descriptor)
    temporary = Path(temp_name)
    index_path = root / INDEX_FILENAME
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE episodes (
                    episode_id TEXT PRIMARY KEY, date TEXT, metadata_json TEXT NOT NULL);
                CREATE TABLE blocks (
                    block_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, date TEXT,
                    start_s REAL NOT NULL, end_s REAL NOT NULL, kind TEXT NOT NULL,
                    record_path TEXT NOT NULL, similarity_eligible INTEGER NOT NULL,
                    similarity_exclusion_reason TEXT, q_vector_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL);
                CREATE TABLE block_events (
                    block_id TEXT NOT NULL, event TEXT NOT NULL, PRIMARY KEY (block_id, event));
                CREATE INDEX blocks_episode ON blocks(episode_id);
                CREATE INDEX blocks_date ON blocks(date);
                CREATE INDEX blocks_time ON blocks(start_s, end_s);
                CREATE INDEX events_event ON block_events(event);
            """)
            db.executemany("INSERT INTO episodes VALUES (?, ?, ?)", episode_rows)
            db.executemany("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", block_rows)
            db.executemany("INSERT INTO block_events VALUES (?, ?)", event_rows)
            db.executemany("INSERT INTO metadata VALUES (?, ?)", [
                ("schema_version", "1"),
                ("manifest_sha256", hashlib.sha256(manifest_bytes).hexdigest()),
                ("blocks_sha256", hashlib.sha256(blocks_bytes).hexdigest()),
                ("neighbor_engine", "sklearn.neighbors.NearestNeighbors"),
                ("distance_metric", "euclidean_unscaled_joint_radians"),
            ])
        os.replace(temporary, index_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "index_path": str(index_path), "episode_count": len(episode_rows),
        "block_count": len(block_rows), "similarity_eligible_count": eligible_count,
        "similarity_excluded_count": len(block_rows) - eligible_count,
        "neighbor_engine": "sklearn.neighbors.NearestNeighbors",
        "distance_metric": "euclidean_unscaled_joint_radians",
    }


def query(library_root, *, joint_positions=None, episode_id=None, date=None,
          event=None, at_s=None, before_monotonic_ns=None, k=5) -> list[dict]:
    """Filter with SQLite, optionally rank finite q6 states with sklearn.

    Metadata filters combine with AND. at_s matches closed [start_s, end_s].
    Joint ties use block_id; metadata order is episode_id/start_s/block_id.
    Missing q blocks remain retrievable when joint_positions is omitted.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 1000:
        raise ValueError("k must be an integer between 1 and 1000")
    if before_monotonic_ns is not None:
        if type(before_monotonic_ns) is not int or before_monotonic_ns < 0:
            raise ValueError('before_monotonic_ns must be an exact nonnegative integer')
        if not isinstance(episode_id,str) or not episode_id:
            raise ValueError('An explicit episode_id is required for a monotonic-clock cutoff')
    joint_vector = _q6(joint_positions) if joint_positions is not None else None
    time_s = _nonnegative_seconds(at_s, "at_s") if at_s is not None else None
    root = Path(library_root).resolve()
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Retrieval index missing: {index_path}; call build_index first")
    conditions, parameters = [], []
    for column, value in (("episode_id", episode_id), ("date", date)):
        if value is not None:
            conditions.append(f"b.{column} = ?")
            parameters.append(value)
    if event is not None:
        conditions.append("EXISTS (SELECT 1 FROM block_events e WHERE e.block_id=b.block_id AND e.event=?)")
        parameters.append(event)
    if time_s is not None:
        conditions.extend(["b.start_s <= ?", "b.end_s >= ?"])
        parameters.extend([time_s, time_s])
    if joint_vector is not None:
        conditions.append("b.similarity_eligible=1")
    if before_monotonic_ns is not None:
        conditions.append("json_type(b.metadata_json, '$.end_ns') = 'integer'")
        conditions.append("json_extract(b.metadata_json, '$.end_ns') <= ?")
        parameters.append(before_monotonic_ns)
    sql = "SELECT b.* FROM blocks b"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY b.episode_id, b.start_s, b.block_id"
    if joint_vector is None:
        sql += " LIMIT ?"
        parameters.append(k)
    # mode=ro prevents an accidental empty database on a misspelled path.
    with closing(sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(sql, parameters).fetchall()
    distances = [None] * len(rows)
    if joint_vector is not None and rows:
        from sklearn.neighbors import NearestNeighbors

        vectors = np.asarray([json.loads(row["q_vector_json"]) for row in rows], dtype=float)
        neighbors = NearestNeighbors(metric="euclidean", algorithm="brute")
        neighbors.fit(vectors)
        # Request all candidates so ties at the kth boundary are resolved by a
        # stable source identifier, independent of sklearn's tied index order.
        native_distances, native_indices = neighbors.kneighbors(
            joint_vector.reshape(1, 6), n_neighbors=len(rows))
        ranked = sorted(zip(native_distances[0], native_indices[0]),
                        key=lambda item: (float(item[0]), rows[int(item[1])]["block_id"]))[:k]
        rows, distances = ([rows[int(i)] for _, i in ranked],
                           [float(distance) for distance, _ in ranked])
    results = []
    for row, distance in zip(rows, distances):
        result = json.loads(row["metadata_json"])
        if not row["similarity_eligible"] and result.get("q_start") is not None:
            # Preserve malformed source evidence as text rather than emitting
            # NaN/Infinity as invalid JSON or exposing it as a usable q6 state.
            result["q_start_source_json"] = json.dumps(result["q_start"])
            result["q_start"] = None
        absolute = _record_path(root, row["record_path"])
        result.update({
            "distance": distance, "distance_metric": "euclidean_unscaled_joint_radians" if distance is not None else None,
            "record_absolute_path": str(absolute), "record_available": absolute.is_file(),
            "similarity_eligible": bool(row["similarity_eligible"]),
            "similarity_exclusion_reason": row["similarity_exclusion_reason"],
        })
        results.append(result)
    return results
