import sys
from pathlib import Path

# let tests import project packages (scripts/, later memory_engine/) from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
