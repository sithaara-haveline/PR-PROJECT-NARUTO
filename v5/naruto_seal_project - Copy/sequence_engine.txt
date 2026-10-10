"""
sequence_engine.py
------------------
All the "remember the seals and match a jutsu" logic, with NO camera / OpenCV
in it, so it can be tested on its own (see test_sequence_engine.py).

Pieces
------
SealStabilizer  turns noisy per-frame predictions into ONE clean event per
                deliberate seal ("Boar was held for 0.7 s").
JutsuBook       jutsu name -> seal sequence, loaded from jutsu_config.json,
                validated against the classes your model knows.
SequenceMachine IDLE -> RECORDING -> RESULT state machine driven by three
                CONTROL seals: start / undo / cast.
DwellButton     "hover for N seconds to press" logic for the on-screen button.
"""

import json
from collections import Counter, deque

IDLE, RECORDING, RESULT = "IDLE", "RECORDING", "RESULT"


# --------------------------------------------------------------------------- #
class SealStabilizer:
    """
    Feed it one (label, confidence) per frame; it returns a label exactly once
    when that label has been the stable majority for `hold_s` seconds.

    To register the SAME seal twice in a row you must break the pose in between
    (relax to Neutral / drop your hands) - otherwise holding one seal would
    register it over and over.
    """

    def __init__(self, window=9, hold_s=0.7, min_conf=0.6, majority=0.6):
        self.buf = deque(maxlen=window)
        self.hold_s, self.min_conf, self.majority = hold_s, min_conf, majority
        self.current = None
        self.since = 0.0
        self.fired = False

    def stable_label(self):
        """Current majority label (or None). Used for UI highlighting."""
        return self.current

    def hold_progress(self, t):
        if self.current is None or self.fired:
            return 0.0
        return min(1.0, (t - self.since) / self.hold_s)

    def update(self, label, conf, t):
        self.buf.append(label if (label is not None and conf >= self.min_conf) else None)
        top, n = Counter(self.buf).most_common(1)[0]
        majority = top if n >= self.majority * self.buf.maxlen else None

        if majority != self.current:
            self.current, self.since, self.fired = majority, t, False

        if self.current is not None and not self.fired and t - self.since >= self.hold_s:
            self.fired = True
            return self.current
        return None

    def reset(self):
        self.buf.clear()
        self.current, self.fired = None, False


# --------------------------------------------------------------------------- #
class JutsuBook:
    def __init__(self, entries):
        self.entries = entries  # list of dicts: name, seals, theme, note

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg["jutsu"])

    def match(self, seq):
        for e in self.entries:
            if list(e["seals"]) == list(seq):
                return e
        return None

    def candidates(self, seq):
        """Jutsu whose sequence STARTS with what has been recorded so far."""
        n = len(seq)
        return [e for e in self.entries if list(e["seals"][:n]) == list(seq)]

    def validate(self, known_classes, controls, idle_labels):
        """Return a list of human-readable problems (empty list = all good)."""
        problems, seen = [], {}
        control_set = set(controls.values())
        for e in self.entries:
            for s in e["seals"]:
                if s not in known_classes:
                    problems.append(f"'{e['name']}' uses '{s}', which your model was not trained on")
                if s in control_set:
                    problems.append(f"'{e['name']}' uses '{s}', which is a CONTROL seal (start/undo/cast)")
                if s in idle_labels:
                    problems.append(f"'{e['name']}' uses the neutral class '{s}'")
            key = tuple(e["seals"])
            if key in seen:
                problems.append(f"'{e['name']}' and '{seen[key]}' have the same sequence")
            seen[key] = e["name"]
        for role, s in controls.items():
            if s not in known_classes:
                problems.append(f"control '{role}' = '{s}' is not a class your model knows")
        if len(set(controls.values())) < len(controls):
            problems.append("two controls share the same seal")
        return problems


# --------------------------------------------------------------------------- #
class SequenceMachine:
    def __init__(self, book, controls, max_len=10, result_s=5.0, timeout_s=40.0):
        self.book, self.controls = book, controls
        self.max_len, self.result_s, self.timeout_s = max_len, result_s, timeout_s
        self.state = IDLE
        self.seq = []
        self.result = None  # matched jutsu entry, or None for "no match"
        self.result_until = 0.0
        self.last_activity = 0.0
        self.toast = ("", 0.0)  # (message, expires_at)

    def _say(self, msg, t, dur=2.0):
        self.toast = (msg, t + dur)

    def message(self, t):
        return self.toast[0] if t < self.toast[1] else ""

    def clear(self, t=0.0):
        self.state, self.seq, self.result = IDLE, [], None
        self._say("Cleared", t, 1.0)

    def tick(self, t):
        """Call every frame. Handles result display end and recording timeout."""
        if self.state == RESULT and t >= self.result_until:
            self.state, self.seq, self.result = IDLE, [], None
        elif self.state == RECORDING and t - self.last_activity > self.timeout_s:
            self.state, self.seq = IDLE, []
            self._say("Timed out - sequence discarded", t, 3.0)

    def handle_seal(self, seal, t):
        """
        Register one deliberate seal. Returns an event string:
        'started' | 'added' | 'undo' | 'cast' | 'nomatch' | 'empty' | 'full' | 'ignored'
        """
        c = self.controls
        if self.state == RESULT:
            return "ignored"

        if self.state == IDLE:
            if seal == c["start"]:
                self.state, self.seq, self.last_activity = RECORDING, [], t
                self._say("Recording started", t)
                return "started"
            return "ignored"

        # ---- RECORDING ----
        self.last_activity = t
        if seal == c["start"]:
            return "ignored"
        if seal == c["undo"]:
            if self.seq:
                removed = self.seq.pop()
                self._say(f"Removed {removed}", t)
                return "undo"
            self._say("Nothing to remove", t)
            return "ignored"
        if seal == c["cast"]:
            if not self.seq:
                self._say("Empty sequence", t)
                return "empty"
            match = self.book.match(self.seq)
            self.state, self.result, self.result_until = RESULT, match, t + self.result_s
            if match is None:
                self._say("No jutsu matches that sequence", t, self.result_s)
                return "nomatch"
            return "cast"
        if len(self.seq) >= self.max_len:
            self._say("Sequence full - cast or undo", t)
            return "full"
        self.seq.append(seal)
        return "added"


# --------------------------------------------------------------------------- #
class DwellButton:
    """Press by keeping the pointer inside the button for `dwell_s` seconds.
    After a press it is latched until the pointer leaves, so one long hover
    can't press it twice."""

    def __init__(self, dwell_s=7.0, grace_s=0.4):
        self.dwell_s, self.grace_s = dwell_s, grace_s
        self.enter_t = None
        self.last_inside_t = 0.0
        self.armed = True

    def progress(self, t):
        return 0.0 if self.enter_t is None else min(1.0, (t - self.enter_t) / self.dwell_s)

    def update(self, inside, t):
        """Returns True on the frame the button is pressed."""
        if inside:
            self.last_inside_t = t
            if not self.armed:
                return False
            if self.enter_t is None:
                self.enter_t = t
            if t - self.enter_t >= self.dwell_s:
                self.enter_t, self.armed = None, False
                return True
        elif t - self.last_inside_t > self.grace_s:
            self.enter_t = None
            self.armed = True
        return False


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
