#!/usr/bin/env python3
"""Print the AgentX measured window as 'START END' (UTC ISO, no zone) from the kit's runner.log.

AIPerf logs only the time of day ("09:24:36.721 NOTICE Phase profiling (profiling) started | ... target: 3600.0s
duration"); the date comes from when agentx.sh was started, rolled over midnight if the time of day is earlier.
Usage: kit_window.py RUNNER_LOG STARTED_ISO
"""

import re
import sys
from datetime import datetime, timedelta

PHASE = re.compile(r"(\d\d:\d\d:\d\d)\.\d+ NOTICE\s+Phase profiling \(profiling\) started.*target: ([0-9.]+)s duration")


def main() -> int:
    started = datetime.fromisoformat(sys.argv[2].rstrip("Z"))
    with open(sys.argv[1], errors="replace") as f:
        for line in f:
            m = PHASE.search(line)
            if m:
                t = datetime.combine(started.date(), datetime.strptime(m.group(1), "%H:%M:%S").time())
                if t < started - timedelta(minutes=5):
                    t += timedelta(days=1)
                end = t + timedelta(seconds=float(m.group(2)))
                print(t.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))
                return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
