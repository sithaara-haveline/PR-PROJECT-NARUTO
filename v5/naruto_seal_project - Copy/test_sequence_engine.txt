"""
test_sequence_engine.py  -  run:  python test_sequence_engine.py
Checks the sequence logic with a fake stream of predictions (no camera needed).
"""

from sequence_engine import (IDLE, RECORDING, RESULT, DwellButton, JutsuBook,
                             SealStabilizer, SequenceMachine, load_config)

cfg = load_config("jutsu_config.json")
book = JutsuBook.from_config(cfg)
ctl = cfg["controls"]
FPS = 30


def make():
    return SequenceMachine(book, ctl, max_len=cfg["timing"]["max_len"],
                           result_s=cfg["timing"]["result_s"], timeout_s=cfg["timing"]["timeout_s"])


def test_config_valid():
    known = {"Boar", "Dog", "Hare", "Horse", "Normal/Neutral", "Ram", "Rat", "Special Cross Seal"}
    assert book.validate(known, ctl, set(cfg["idle_labels"])) == []


def test_happy_path():
    m, t = make(), 0.0
    assert m.handle_seal("Ram", t) == "ignored"            # ignored before START
    assert m.handle_seal(ctl["start"], t) == "started"
    for s in ["Ram", "Horse", "Boar"]:
        assert m.handle_seal(s, t) == "added"
    assert m.handle_seal(ctl["cast"], t) == "cast"
    assert m.state == RESULT and m.result["name"] == "Fireball Jutsu"
    assert m.handle_seal("Ram", t) == "ignored"            # ignored while result is showing
    m.tick(t + 100)
    assert m.state == IDLE and m.seq == []


def test_undo_and_wrong_sequence():
    m, t = make(), 0.0
    m.handle_seal(ctl["start"], t)
    for s in ["Ram", "Hare"]:
        m.handle_seal(s, t)
    assert m.handle_seal(ctl["undo"], t) == "undo" and m.seq == ["Ram"]
    m.handle_seal("Boar", t)
    assert m.handle_seal(ctl["cast"], t) == "cast" and m.result["name"] == "Shadow Clone Jutsu"

    m2 = make()
    m2.handle_seal(ctl["start"], t)
    m2.handle_seal("Hare", t)
    assert m2.handle_seal(ctl["cast"], t) == "nomatch" and m2.result is None


def test_edge_cases():
    m, t = make(), 0.0
    m.handle_seal(ctl["start"], t)
    assert m.handle_seal(ctl["cast"], t) == "empty"
    assert m.handle_seal(ctl["undo"], t) == "ignored"      # nothing to remove
    assert m.handle_seal(ctl["start"], t) == "ignored"     # START again while recording
    for _ in range(cfg["timing"]["max_len"]):
        m.handle_seal("Ram", t)
    assert m.handle_seal("Ram", t) == "full"
    m.tick(t + cfg["timing"]["timeout_s"] + 1)
    assert m.state == IDLE                                  # timeout discards


def test_stabilizer_one_event_per_hold():
    st, events, t = SealStabilizer(window=9, hold_s=0.7, min_conf=0.6), [], 0.0
    stream = ([("Ram", 0.9)] * 40 + [("Normal/Neutral", 0.9)] * 20 +   # hold Ram, relax
              [("Ram", 0.9)] * 40 +                                     # Ram again -> 2nd event
              [("Boar", 0.3)] * 40 +                                    # low confidence -> nothing
              [("Boar", 0.9), ("Hare", 0.9)] * 20)                      # flicker -> nothing
    for lab, conf in stream:
        e = st.update(lab, conf, t)
        if e:
            events.append(e)
        t += 1 / FPS
    assert events.count("Ram") == 2, events
    assert "Boar" not in events and "Hare" not in events, events


def test_dwell_button():
    b, t, pressed = DwellButton(dwell_s=7.0, grace_s=0.4), 0.0, 0
    for _ in range(int(9 * FPS)):                           # 9 s inside -> one press only
        pressed += b.update(True, t)
        t += 1 / FPS
    assert pressed == 1
    for _ in range(int(0.6 * FPS)):                         # leave -> re-arm
        b.update(False, t)
        t += 1 / FPS
    for _ in range(int(3 * FPS)):                           # 3 s is not enough
        pressed += b.update(True, t)
        t += 1 / FPS
    assert pressed == 1
    b2, t = DwellButton(dwell_s=7.0, grace_s=0.4), 0.0      # flicker shorter than grace keeps progress
    for i in range(int(8 * FPS)):
        inside = not (i % 45 == 0 and i > 0)               # 1-frame dropout every 1.5 s
        if b2.update(inside, t):
            break
        t += 1 / FPS
    assert t < 7.5, t


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
