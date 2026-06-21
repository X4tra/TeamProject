"""
tracker.py  —  Simple IoU-based multi-object tracker
=====================================================

Matches new detections to existing tracks using Intersection-over-Union (IoU).
No external dependencies beyond numpy.

Interface expected by utils.py:
    tracker = Tracker(iou_threshold=0.25, max_age=5, min_hits=2)
    tracked = tracker.update(boxes)   # boxes = [[x1,y1,x2,y2], ...]
    # tracked is a list of [x1, y1, x2, y2, obj_id]
"""

import numpy as np


def _iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class _Track:
    """Internal state for a single tracked object."""

    _next_id = 1

    def __init__(self, box: list):
        self.id       = _Track._next_id
        _Track._next_id += 1
        self.box      = box          # [x1, y1, x2, y2]
        self.age      = 1            # frames since first seen
        self.hits     = 1            # consecutive matched frames
        self.no_match = 0            # consecutive unmatched frames

    def update(self, box: list) -> None:
        self.box      = box
        self.hits    += 1
        self.no_match = 0
        self.age     += 1

    def mark_missed(self) -> None:
        self.no_match += 1
        self.age      += 1


class Tracker:
    """
    IoU-based multi-object tracker.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU required to match a detection to an existing track.
    max_age : int
        How many consecutive unmatched frames before a track is deleted.
    min_hits : int
        How many consecutive matched frames before a track is returned
        as a confirmed detection.
    """

    def __init__(self, iou_threshold: float = 0.25,
                 max_age: int = 5, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.min_hits      = min_hits
        self._tracks: list[_Track] = []

    def update(self, boxes: list[list]) -> list[list]:
        """
        Match *boxes* (list of [x1,y1,x2,y2]) to existing tracks.

        Returns
        -------
        list of [x1, y1, x2, y2, obj_id]  for every confirmed track.
        """
        # ------------------------------------------------------------------ #
        # 1. Match detections → tracks via greedy IoU                         #
        # ------------------------------------------------------------------ #
        matched_track_idx  = set()
        matched_detect_idx = set()

        if self._tracks and boxes:
            # Build IoU matrix  (tracks × detections)
            iou_matrix = np.zeros((len(self._tracks), len(boxes)))
            for t, track in enumerate(self._tracks):
                for d, box in enumerate(boxes):
                    iou_matrix[t, d] = _iou(track.box, box)

            # Greedy: repeatedly pick the highest IoU pair
            while True:
                t, d = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[t, d] < self.iou_threshold:
                    break
                matched_track_idx.add(t)
                matched_detect_idx.add(d)
                self._tracks[t].update(boxes[d])
                iou_matrix[t, :] = -1   # prevent re-matching this track
                iou_matrix[:, d] = -1   # prevent re-matching this detection

        # ------------------------------------------------------------------ #
        # 2. Mark unmatched tracks as missed                                  #
        # ------------------------------------------------------------------ #
        for t, track in enumerate(self._tracks):
            if t not in matched_track_idx:
                track.mark_missed()

        # ------------------------------------------------------------------ #
        # 3. Create new tracks for unmatched detections                       #
        # ------------------------------------------------------------------ #
        for d, box in enumerate(boxes):
            if d not in matched_detect_idx:
                self._tracks.append(_Track(box))

        # ------------------------------------------------------------------ #
        # 4. Delete stale tracks                                              #
        # ------------------------------------------------------------------ #
        self._tracks = [t for t in self._tracks
                        if t.no_match <= self.max_age]

        # ------------------------------------------------------------------ #
        # 5. Return confirmed tracks only                                      #
        # ------------------------------------------------------------------ #
        results = []
        for track in self._tracks:
            if track.hits >= self.min_hits and track.no_match == 0:
                x1, y1, x2, y2 = track.box
                results.append([x1, y1, x2, y2, track.id])

        return results
