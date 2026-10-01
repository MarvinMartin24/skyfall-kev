#!/usr/bin/env python3
"""Play Skyfall from any directory:  python3 /path/to/skyfall-kev/skyfall.py
Same flags as game.play (defaults to --policy human)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game.play import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["--policy", "human"]
    main()
