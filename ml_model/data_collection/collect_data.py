"""
collect_data.py
───────────────
Reads labeled vibration + temperature data from the STM32 over serial
(the node has a special DATA_COLLECT mode that streams raw CSV).

Usage:
    python collect_data.py --port COM3 --label normal --duration 60
    python collect_data.py --port COM3 --label bearing_fault --duration 60

Output:
    data/<label>_<timestamp>.csv
"""

import argparse
import csv
import os
import time
import serial
from datetime import datetime

COLUMNS = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "temp", "label"]

def parse_args():
    parser = argparse.ArgumentParser(description="Sensor data collector")
    parser.add_argument("--port",     required=True, help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud",     type=int, default=115200)
    parser.add_argument("--label",    required=True,
                        choices=["normal", "bearing_fault", "imbalance", "looseness"])
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds")
    parser.add_argument("--output",   default="data", help="Output directory")
    return parser.parse_args()

def collect(args):
    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(args.output, f"{args.label}_{ts}.csv")

    with serial.Serial(args.port, args.baud, timeout=1) as ser, \
         open(filepath, "w", newline="") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()

        print(f"[INFO] Collecting '{args.label}' data for {args.duration}s → {filepath}")
        ser.write(b"DATA_MODE\r\n")   # Put node in streaming mode
        time.sleep(0.5)

        deadline = time.time() + args.duration
        sample_count = 0

        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line.startswith("D:"):
                continue
            # Format: D:timestamp,ax,ay,az,gx,gy,gz,temp
            try:
                parts = line[2:].split(",")
                if len(parts) != 8:
                    continue
                writer.writerow({
                    "timestamp_ms": int(parts[0]),
                    "ax":    float(parts[1]),
                    "ay":    float(parts[2]),
                    "az":    float(parts[3]),
                    "gx":    float(parts[4]),
                    "gy":    float(parts[5]),
                    "gz":    float(parts[6]),
                    "temp":  float(parts[7]),
                    "label": args.label,
                })
                sample_count += 1
                if sample_count % 500 == 0:
                    print(f"  {sample_count} samples collected...")
            except (ValueError, IndexError) as e:
                pass  # Skip malformed lines

        print(f"[DONE] {sample_count} samples saved to {filepath}")
        ser.write(b"NORMAL_MODE\r\n")

if __name__ == "__main__":
    collect(parse_args())