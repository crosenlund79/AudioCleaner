import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from audiorefine.gui.app import main

    sys.exit(main())
