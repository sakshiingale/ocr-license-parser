#!/usr/bin/env python3
"""
aggregator.py
- Reads extraction_logs.jsonl and computes per-minute throughput and average parse time.
- Outputs a CSV or prints a textual report.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import argparse

LOG_PATH = Path("extraction_logs.jsonl")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

def minute_key(ts_str):
    # returns 'YYYY-MM-DDTHH:MM' as key
    dt = datetime.strptime(ts_str, TIME_FORMAT)
    return dt.strftime("%Y-%m-%dT%H:%M")

def aggregate(log_path=LOG_PATH):
    per_min = defaultdict(list)  # minute -> list of durations (ms)
    total = 0
    succeeded = 0

    with open(log_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            # use end_ts or start_ts as the bucket - using end_ts is common
            mk = minute_key(obj["end_ts"])
            per_min[mk].append(obj["duration_ms"])
            total += 1
            if obj["status"] == "success":
                succeeded += 1

    # Build rows
    rows = []
    for minute in sorted(per_min.keys()):
        durations = per_min[minute]
        count = len(durations)
        avg_ms = sum(durations) / count if count else 0
        rows.append((minute, count, avg_ms))

    return rows, total, succeeded

def print_report(rows, total, succeeded, top_n=20):
    print("Total processed:", total, "Succeeded:", succeeded)
    print("Minute, docs_count, avg_duration_ms")
    for minute, count, avg in rows[-top_n:]:
        print(f"{minute}, {count}, {avg:.1f}")

def main():
    rows, total, succeeded = aggregate()
    print_report(rows, total, succeeded)

if __name__ == "__main__":
    main()
