import json
import numpy as np
import pathlib


def _to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.void):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


data_dir = pathlib.Path(__file__).resolve().parents[2] / "data"

npz_files = sorted(data_dir.rglob("*.npz"))
print(f"Found {len(npz_files)} .npz files\n")

for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)

    payload = {}
    for key in data.files:
        val = data[key]
        # allow_pickle can produce 0-d object arrays wrapping dicts/lists
        if isinstance(val, np.ndarray) and val.ndim == 0:
            val = val.item()
        payload[key] = _to_serializable(val)

    json_path = npz_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_to_serializable)

    print(f"✓ {npz_path.relative_to(data_dir)}  →  {json_path.name}")

