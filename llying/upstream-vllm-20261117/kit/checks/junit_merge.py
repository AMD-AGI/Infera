#!/usr/bin/env python3
"""Merge per-file pytest junit XMLs into one (for runs where each test file gets its own process).
  junit_merge.py OUT.xml PART.xml...
A part that is missing (its process died before writing it) becomes one failed case named after the file, so the
comparison sees it.
"""

import sys
import xml.etree.ElementTree as ET

out, parts = sys.argv[1], sys.argv[2:]
suite = ET.Element("testsuite", name="merged")
for part in parts:
    try:
        root = ET.parse(part).getroot()
    except (OSError, ET.ParseError):
        case = ET.SubElement(suite, "testcase", classname="", name=part.rsplit("/", 1)[-1])
        ET.SubElement(case, "failure", message="process died before writing its results")
        continue
    for case in root.iter("testcase"):
        suite.append(case)
ET.ElementTree(ET.Element("testsuites")).getroot().append(suite)
tree = ET.ElementTree(suite)
tree.write(out)
