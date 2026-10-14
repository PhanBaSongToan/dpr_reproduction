
import gzip
from pathlib import Path

files = sorted(Path("data/raw").rglob("nq-*.tmp"))

for path in files:
    print(f"\nChecking: {path.name}", flush=True)
    total = 0

    try:
        with gzip.open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)

        print(f"VALID GZIP: {path.name}")
        print(f"Decompressed size: {total / 1024**2:.2f} MB")
    except (OSError, EOFError, MemoryError) as error:
        print(f"INVALID OR INCOMPLETE: {path.name}")
        print(f"Error: {error}")
