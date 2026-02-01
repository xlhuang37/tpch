#!/usr/bin/env python3
"""
Open-loop ClickHouse workload tester.

This is a wrapper script that runs the test_open package.
Execute from the tpch/ directory:

    python test_open.py --schedules-dir ./schedules/ --queries-dir ./queries/

For full options:
    python test_open.py --help
"""

import asyncio
from test_open.main import main

if __name__ == "__main__":
    asyncio.run(main())
