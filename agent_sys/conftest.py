"""Put ``src/`` on ``sys.path``.

The two-line version of an editable install. ``agent_sys`` is not in the
repository's ``[tool.setuptools.packages.find] include``, and while this is
unreleased it does not need to be.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
