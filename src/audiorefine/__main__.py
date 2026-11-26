"""``python -m audiorefine`` starts the desktop app; ``python -m audiorefine <command>`` runs the engine."""
import sys

if len(sys.argv) > 1:
    from .cli import main

    sys.exit(main())
else:
    from .gui.app import main

    sys.exit(main())
