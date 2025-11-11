import time, sys
def info(msg): print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
def warn(msg): print(f"[WARN] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}", file=sys.stderr)
