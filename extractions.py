#!/usr/bin/env python3
"""
extractions.py
- Walks a folder (or single file) and POSTs each file to the /api/extract-fields endpoint.
- Logs one JSON line per attempt to extraction_logs.jsonl with:
  { request_id, filename, start_ts, end_ts, duration_ms, status, http_status, detail }
"""

import os
import sys
import time
import uuid
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("extraction_logs.jsonl")
API_URL = os.getenv("EXTRACTION_API_URL", "http://localhost:8000/api/extract-fields")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

def log_event(event: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

def process_file(path: Path, timeout=60):
    request_id = str(uuid.uuid4())
    start = time.time()
    start_ts = datetime.utcfromtimestamp(start).strftime(TIME_FORMAT)
    files = {"pdf": (path.name, open(path, "rb"), "application/octet-stream")}
    try:
        resp = requests.post(API_URL, files=files, timeout=timeout)
        end = time.time()
        end_ts = datetime.utcfromtimestamp(end).strftime(TIME_FORMAT)
        duration_ms = int((end - start) * 1000)
        status = "success" if resp.status_code == 200 else "error"
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        event = {
            "request_id": request_id,
            "filename": path.name,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_ms": duration_ms,
            "status": status,
            "http_status": resp.status_code,
            "detail": detail
        }
        log_event(event)
        return event
    except Exception as e:
        end = time.time()
        end_ts = datetime.utcfromtimestamp(end).strftime(TIME_FORMAT)
        duration_ms = int((end - start) * 1000)
        event = {
            "request_id": request_id,
            "filename": path.name,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_ms": duration_ms,
            "status": "exception",
            "http_status": None,
            "detail": str(e)
        }
        log_event(event)
        return event

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="File or directory to process")
    parser.add_argument("--delay", "-d", type=float, default=0.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.is_file()])
    elif input_path.is_file():
        files = [input_path]
    else:
        print("Input path not found:", input_path)
        sys.exit(1)

    for p in files:
        print("Processing", p)
        evt = process_file(p)
        print(" ->", evt["status"], evt["duration_ms"], "ms")
        if args.delay:
            time.sleep(args.delay)

if __name__ == "__main__":
    main()
