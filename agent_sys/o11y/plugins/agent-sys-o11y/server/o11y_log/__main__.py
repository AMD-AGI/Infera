import sys

from o11y_job import main

from . import handle

sys.exit(main("log", handle))
