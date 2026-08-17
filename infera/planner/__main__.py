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

from infera.planner import planner_metrics
from infera.planner.args import PlannerArgs, parse_planner_args
from infera.planner.connectors.base import build_connector
from infera.planner.core import SlaPlanner
from infera.planner.metrics_source import MetricsSource
from infera.planner.perf_model import PerfModel
from infera.planner.profile_data import ProfileDataError, load_profile_data

logger = logging.getLogger(__name__)


async def _run(args: PlannerArgs) -> None:
    perf_model = PerfModel(load_profile_data(args.profile_results))
    metrics_source = MetricsSource(args.metrics_urls)
    connector = build_connector(args)
    planner = SlaPlanner(
        args,
        perf_model,
        metrics_source=metrics_source,
        connector=connector,
    )
    logger.info(
        "scraping %s; actuating via %s%s",
        ", ".join(args.metrics_urls),
        args.connector,
        " (no-operation)" if args.no_operation else "",
    )
    try:
        await planner.run()
    finally:
        await metrics_source.aclose()
        await connector.aclose()


def main() -> None:
    args = parse_planner_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    planner_metrics.serve(args.metrics_port)
    try:
        asyncio.run(_run(args))
    except ProfileDataError as exc:
        # A bad performance model can only produce bad decisions, so refuse to
        # start rather than scale on garbage.
        logger.error("%s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("planner stopped")


if __name__ == "__main__":
    main()
