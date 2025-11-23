import sys
from pathlib import Path


def main():
    # Ensure repository root is on sys.path so imports like "src.api" resolve
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from main import main as run_main

    run_main()
