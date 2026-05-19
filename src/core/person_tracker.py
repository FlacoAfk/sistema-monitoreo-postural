"""
Person Identity Tracker — CentroidTracker

Assigns stable person IDs across frames by matching bounding-box
centroids via nearest-neighbour with a max-distance threshold.

Problem it solves:
    YOLO pose models return detections in arbitrary order each frame.
    Without tracking, ``person_id = p_idx`` changes every frame when
    people move, enter, or leave the scene — causing the mobile app
    to show stale / mismatched person cards.

Algorithm:
    1. For each new detection, compute its centroid (box center).
    2. Compute the distance to every tracked person's last centroid.
    3. Use a greedy nearest-match: closest pair under MAX_DISTANCE
       wins; unmatched detections become new IDs; unmatched trackers
       are marked as "missing" and expire after max_absent_s.
    4. Ghost buffer: persons that expired from active tracking are
       kept in a ghost buffer (wall-clock timed). On re-appearance,
       their ID is restored if matched within MAX_DISTANCE.
    5. Permanent deletion: ghosts that exceed max_absent_s are removed
       and IDs are compacted to fill gaps.

Universidad Surcolombiana, 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class _TrackedPerson:
    """Internal state for one tracked person."""
    person_id: int
    cx: float          # Last known centroid X
    cy: float          # Last known centroid Y
    missing_time: float = 0.0  # Seconds since last seen


@dataclass
class CentroidTracker:
    """Stable multi-person identity tracker using bounding-box centroids.

    Usage::

        tracker = CentroidTracker()
        ids, remap = tracker.update(centroids=[(250, 400), (600, 380)])
        # ids = [0, 1]  ← stable across frames

        # After a few frames, person 0 leaves:
        ids, remap = tracker.update(centroids=[(595, 385)])
        # ids = [1]  ← person 0 moved to ghost buffer

        # Person 0 returns within max_absent_s:
        ids, remap = tracker.update(centroids=[(248, 402)])
        # ids = [0, 1]  ← ID restored from ghost buffer

        # Person 0 permanently left (ghost expired):
        left = tracker.get_left_persons()
        # left = [0]  ← notify the app

    Thread safety:
        NOT thread-safe. Call from the same thread as process_frame()
        (the Gradio generator).
    """

    # Max pixel distance to consider two centroids the same person.
    # At 640x480, 120px ~ 25% of width — generous enough for normal
    # movement between frames (~0.1s at 10 FPS), tight enough to
    # distinguish two people standing side-by-side.
    MAX_DISTANCE: float = 120.0

    # Wall-clock seconds a person can be absent before being
    # permanently forgotten. At 600 s (10 min), this provides ample
    # grace for momentary occlusions while still cleaning up stale
    # identities. Frame-rate independent.
    max_absent_s: float = 600.0

    _next_id: int = field(default=0, init=False)
    _tracked: dict[int, _TrackedPerson] = field(default_factory=dict, init=False)
    _ghost_buffer: dict[int, tuple[tuple[float, float], float]] = field(
        default_factory=dict, init=False
    )
    # Maps person_id -> ((cx, cy), exit_timestamp)
    _left_buffer: list[int] = field(default_factory=list, init=False)
    _last_update_time: float = field(default=0.0, init=False)

    # ── Public API ──────────────────────────────────────────────────────

    def update(
        self, centroids: list[tuple[float, float]], now: float | None = None
    ) -> tuple[list[int], dict[int, int]]:
        """Match new detections to tracked persons and return stable IDs.

        Args:
            centroids: List of (cx, cy) tuples for each detected person
                       in the current frame.
            now: Optional wall-clock timestamp. Defaults to time.time().

        Returns:
            Tuple of (assigned_ids, remap_dict).
            - assigned_ids: Stable person_id per input centroid, same order.
            - remap_dict: {old_id: new_id} if compaction occurred, else {}.
        """
        if now is None:
            now = time.time()

        # Initialise timestamp on first call
        if self._last_update_time == 0.0:
            self._last_update_time = now

        elapsed = max(0.0, now - self._last_update_time)
        self._last_update_time = now

        n_new = len(centroids)

        if n_new == 0:
            # No detections — all tracked persons accrue missing time
            for tp in self._tracked.values():
                tp.missing_time += elapsed
            self._collect_expired(now)
            self._clean_ghosts(now)
            remap = self._compact_ids() if self._left_buffer else {}
            return ([], remap)

        # ── Step 1: Match input centroids to active tracked persons ──
        tracked_ids = list(self._tracked.keys())
        tracked_persons = [self._tracked[tid] for tid in tracked_ids]
        n_tracked = len(tracked_ids)

        used_new: set[int] = set()
        used_tracked: set[int] = set()
        matches: dict[int, int] = {}  # new_idx -> tracked_id

        if n_tracked > 0:
            pairs: list[tuple[float, int, int]] = []
            for i in range(n_new):
                cx_new, cy_new = centroids[i]
                for j in range(n_tracked):
                    tp = tracked_persons[j]
                    dist = math.hypot(cx_new - tp.cx, cy_new - tp.cy)
                    if dist <= self.MAX_DISTANCE:
                        pairs.append((dist, i, j))
            pairs.sort(key=lambda p: p[0])
            for dist, i, j in pairs:
                if i in used_new or j in used_tracked:
                    continue
                matches[i] = tracked_ids[j]
                used_new.add(i)
                used_tracked.add(j)

        # ── Step 2: Unmatched input -> ghost buffer re-ID ────────────
        if self._ghost_buffer:
            ghost_items = list(self._ghost_buffer.items())
            ghost_pairs: list[tuple[float, int, int]] = []
            for i in range(n_new):
                if i in used_new:
                    continue
                cx_new, cy_new = centroids[i]
                for gid, ((gcx, gcy), _) in ghost_items:
                    dist = math.hypot(cx_new - gcx, cy_new - gcy)
                    if dist <= self.MAX_DISTANCE:
                        ghost_pairs.append((dist, i, gid))
            ghost_pairs.sort(key=lambda p: p[0])
            used_ghosts: set[int] = set()
            for dist, i, gid in ghost_pairs:
                if i in used_new or gid in used_ghosts:
                    continue
                # Restore from ghost -> active tracking
                self._ghost_buffer.pop(gid)
                self._tracked[gid] = _TrackedPerson(
                    person_id=gid,
                    cx=centroids[i][0],
                    cy=centroids[i][1],
                    missing_time=0.0,
                )
                matches[i] = gid
                used_new.add(i)
                used_ghosts.add(gid)

        # ── Step 3: Remaining unmatched -> new IDs ───────────────────
        result: list[int] = [0] * n_new
        for i in range(n_new):
            if i in matches:
                tid = matches[i]
                tp = self._tracked[tid]
                tp.cx, tp.cy = centroids[i]
                tp.missing_time = 0.0
                result[i] = tid
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracked[new_id] = _TrackedPerson(
                    person_id=new_id,
                    cx=centroids[i][0],
                    cy=centroids[i][1],
                )
                result[i] = new_id

        # ── Step 4: Unmatched active -> incr missing, ghost if expired ──
        for j in range(n_tracked):
            if j not in used_tracked:
                tid = tracked_ids[j]
                tp = self._tracked[tid]
                tp.missing_time += elapsed
                if tp.missing_time > self.max_absent_s:
                    self._ghost_buffer[tid] = ((tp.cx, tp.cy), now)
                    del self._tracked[tid]

        # ── Step 5: Ghost cleanup (permanent deletion) ──────────────
        self._clean_ghosts(now)

        # ── Step 6: Compaction if any permanent deletion occurred ───
        remap = self._compact_ids() if self._left_buffer else {}

        return (result, remap)

    def get_left_persons(self) -> list[int]:
        """Return the list of person IDs that were PERMANENTLY deleted
        (ghost timed out) since the last call. IDs that transitioned to
        ghost buffer are NOT included. Each ID is returned ONCE.
        """
        left = list(self._left_buffer)
        self._left_buffer.clear()
        return left

    @property
    def active_count(self) -> int:
        """Number of currently tracked (non-expired) persons."""
        return len(self._tracked)

    def reset(self) -> None:
        """Clear all tracking state (e.g., on session start)."""
        self._tracked.clear()
        self._ghost_buffer.clear()
        self._left_buffer.clear()
        self._next_id = 0
        self._last_update_time = 0.0

    # ── Internal ────────────────────────────────────────────────────────

    def _collect_expired(self, now: float) -> None:
        """Move tracked persons whose missing_time exceeds max_absent_s
        into the ghost buffer with the current timestamp.
        """
        expired = [
            tid
            for tid, tp in list(self._tracked.items())
            if tp.missing_time > self.max_absent_s
        ]
        for tid in expired:
            tp = self._tracked[tid]
            self._ghost_buffer[tid] = ((tp.cx, tp.cy), now)
            del self._tracked[tid]

    def _clean_ghosts(self, now: float) -> None:
        """Permanently delete ghosts whose exit timestamp is older than
        max_absent_s. Deleted IDs are appended to _left_buffer.
        """
        expired = [
            gid
            for gid, (_, ts) in list(self._ghost_buffer.items())
            if now - ts > self.max_absent_s
        ]
        for gid in expired:
            del self._ghost_buffer[gid]
            self._left_buffer.append(gid)

    def _compact_ids(self) -> dict[int, int]:
        """Remap all active + ghost IDs to sequential integers starting
        from 1. Returns {old_id: new_id} for every remapped ID, or
        {} if IDs are already sequential.
        """
        all_ids = sorted(set(self._tracked.keys()) | set(self._ghost_buffer.keys()))
        if not all_ids:
            self._next_id = 0
            return {}

        needs_remap = any(old_id != (i + 1) for i, old_id in enumerate(all_ids))
        if not needs_remap:
            return {}

        remap: dict[int, int] = {}
        new_tracked: dict[int, _TrackedPerson] = {}
        new_ghost: dict[int, tuple[tuple[float, float], float]] = {}

        for new_id, old_id in enumerate(all_ids, start=1):
            remap[old_id] = new_id
            if old_id in self._tracked:
                tp = self._tracked[old_id]
                tp.person_id = new_id
                new_tracked[new_id] = tp
            if old_id in self._ghost_buffer:
                new_ghost[new_id] = self._ghost_buffer[old_id]

        self._tracked = new_tracked
        self._ghost_buffer = new_ghost
        self._next_id = (len(all_ids) if all_ids else 0) + 1
        return remap
