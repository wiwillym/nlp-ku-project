import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = os.path.join(PROJECT_ROOT, "data")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Data directory not found at {DATA_PATH}. Please ensure the data is placed in the correct location."
    )
