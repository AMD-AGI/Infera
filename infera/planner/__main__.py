###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entry point for the SLA planner: ``python -m infera.planner``.

Runs as its own long-lived process, like ``infera.kvd`` and ``infera.gaie``, so
a planner restart or crash never touches request serving.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from infera.planner.args import PlannerArgs, parse_planner_args
from infera.planner.core import SlaPlanner
from infera.planner.metrics_source import MetricsSource
from infera.planner.perf_model import PerfModel
from infera.planner.profile_data import ProfileDataError, load_profile_data

logger = logging.getLogger(__name__)


async def _run(args: PlannerArgs) -> None:
    perf_model = PerfModel(load_profile_data(args.profile_results))
    metrics_source = MetricsSource(args.metrics_urls, model=args.model, router="disagg")
    planner = SlaPlanner(
        args,
        perf_model,
        metrics_source=metrics_source,
    )
    logger.info("scraping %s", ", ".join(args.metrics_urls))
    try:
        await planner.run()
    finally:
        await metrics_source.aclose()


def main() -> None:
    args = parse_planner_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except ProfileDataError as exc:
        # A bad performance model can only produce bad decisions, so refuse to
        # start rather than decide on garbage.
        logger.error("%s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("planner stopped")


if __name__ == "__main__":
    main()
