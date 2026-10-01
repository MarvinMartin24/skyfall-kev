#!/usr/bin/env python3
"""Play Skyfall in the terminal, from any directory:  python3 /path/to/skyfall-kev/skyfall.py [--policy oracle|model] [--fps 3] [--surprise]
Keys: a/← left, d/→ right, enter stay, space open, h hint, r replay, q quit."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game.tui import main  # noqa: E402

if __name__ == "__main__":
    main()
