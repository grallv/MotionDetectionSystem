import argparse, ast, csv, math, os
import statistics as stats
from typing import Dict, Tuple, Any

TOL_SEC = 0.10  # 100 ms tolerance
TOL_FRAMES = 1  # ±1 frame tolerance

def parse_float(s):
    try:
        return float(ast.literal_eval(s)) if s else None
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def load_events(path="event_log.csv"):
    if not os.path.exists(path):
        raise SystemExit("event_log.csv not found. Run with the instrumentation enabled.")
    starts, ends = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = int(row["event_id"]) if row["event_id"] else None
            if row["type"] == "start":
                starts[eid] = row
            elif row["type"] == "end":
                ends[eid] = row
    return starts, ends

def check_event(start, end, tol_frames, tol_sec):
    # Parse fields
    fps = float(start["fps"])
    buffer_s = float(start["buffer_s"])
    cooldown_s = float(start["cooldown_s"])
    pre_roll_frames = int(start["pre_roll_frames"])
    frames_written = int(end["frames_written"])
    filename = start["filename"]

    # time stamps
    t_trigger = parse_float(start.get("t_trigger"))
    t_end = parse_float(end.get("t_end"))
    t_last_motion = parse_float(end.get("t_last_motion"))

    # Expected pre-roll frames
    exp_pre = round(buffer_s * fps)
    pre_ok = abs(pre_roll_frames - exp_pre) <= TOL_FRAMES

    # Tail check by TIME, not frames:
    # Tail duration is from last_motion_time to end.
    # If last_motion_time wasn't updated after trigger, it equals t_trigger.
    tail_s = None
    tail_ok = False

    if t_end is not None and t_last_motion is not None:
        tail_s = t_end - t_last_motion
        tail_ok = (tail_s + TOL_SEC) >= cooldown_s
    else:
        # Fallback: approximate tail via frames/fps if timestamps are missing
        tail_s = frames_written / fps if fps > 0 else 0.0
        tail_ok = (tail_s + TOL_SEC) >= cooldown_s

    # REalized FPS during recording (post-trigger only)
    realized_fps = None
    rec_s = None
    if t_end is not None and t_trigger is not None:
        rec_s = t_end - t_trigger
        if rec_s > 0:
            realized_fps = frames_written / rec_s

    passed = bool(pre_ok and tail_ok)

    return {
        "filename": filename,
        "fps_nominal": fps,
        "pre_roll_frames": pre_roll_frames,
        "exp_pre_frames": exp_pre,
        "tail_seconds": tail_s,
        "exp_tail_seconds": cooldown_s,
        "frames_written": frames_written,
        "realized_fps": realized_fps,
        "recording_seconds": rec_s,
        "pre_roll_ok": pre_ok,
        "tail_ok": tail_ok,
        "pass": passed,
    }

def fmt(x, nd=2, fallback="n/a"):
    if x is None:
        return fallback
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def main():
    ap = argparse.ArgumentParser(description="Validate motion-event clips using event_log.csv")
    ap.add_argument("--log", default="event_log.csv", help="Path to event log CSV (default: event_log.csv)")
    ap.add_argument("--tol-sec", type=float, default=0.10, help="Tail time tolerance in seconds (default: 0.10)")
    ap.add_argument("--tol-frames", type=int, default=1, help="Pre-roll tolerance in frames (default: 1)")
    ap.add_argument("--stats", action="store_true", help="Print aggregate statistics")
    args = ap.parse_args()

    starts, ends = load_events(args.log)

    print("Event Validation Report")
    print("=" * 80)

    if not starts:
        print("[ERROR] No start events found.")
        return

    # Determine event ID ordering (by numeric id ascending)
    eids = sorted(starts.keys())
    overall_pass = True
    have_fail = False
    missing_end = []

    # For stats
    pre_list, tail_list, fps_real_list = [], [], []

    for eid in eids:
        start = starts[eid]
        end = ends.get(eid)

        if end is None:
            print(f"[WARN] Event {eid} has no 'end' row; clip may have been interrupted.")
            overall_pass = False
            missing_end.append(eid)
            continue

        res = check_event(start, end, tol_frames=args.tol_frames, tol_sec=args.tol_sec)

        print(f"- {res['filename']}")
        print(
            f"  fps_nominal={fmt(res['fps_nominal'], 2)}  "
            f"realized_fps={fmt(res['realized_fps'], 2)}  "
            f"recording_s={fmt(res['recording_seconds'], 2)}"
        )
        print(
            f"  pre_roll={res['pre_roll_frames']} (exp {res['exp_pre_frames']})  "
            f"tail_s={fmt(res['tail_seconds'], 2)} (min {fmt(res['exp_tail_seconds'], 2)})"
        )
        print(
            f"  pre_roll_ok={res['pre_roll_ok']}  tail_ok={res['tail_ok']}  PASS={res['pass']}"
        )

        if res["pass"] is False:
            have_fail = True
            overall_pass = False

        # Collect stats if available
        if res["pre_roll_frames"] is not None:
            pre_list.append(res["pre_roll_frames"])
        if res["tail_seconds"] is not None:
            tail_list.append(res["tail_seconds"])
        if res["realized_fps"] is not None:
            fps_real_list.append(res["realized_fps"])

    print("=" * 80)

    if args.stats:
        def mean_sd(a):
            if not a:
                return ("n/a", "n/a")
            if len(a) == 1:
                return (fmt(a[0], 2), "n/a")
            return (fmt(stats.mean(a), 2), fmt(stats.pstdev(a), 2))

        m_pre, sd_pre = mean_sd(pre_list)
        m_tail, sd_tail = mean_sd(tail_list)
        m_fps, sd_fps = mean_sd(fps_real_list)

        print("Aggregate Statistics")
        print(f"  Pre-roll frames: mean={m_pre}  sd={sd_pre}")
        print(f"  Tail seconds:    mean={m_tail}  sd={sd_tail}")
        print(f"  Realized FPS:    mean={m_fps}   sd={sd_fps}")
        print("-" * 80)

    status = "PASS" if overall_pass else "CHECK FAILURES ABOVE"
    print("OVERALL:", status)

    if missing_end:
        print(f"[NOTE] Missing 'end' for event IDs: {missing_end}. "
              f"Ensure the program writes an 'end' row on cooldown stop and on shutdown.")

if __name__ == "__main__":
    main()
