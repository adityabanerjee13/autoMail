"""Test-wide setup.

psycopg3's async mode refuses to run on Python's default Windows event loop
(ProactorEventLoop). Production is Linux, where this is a no-op, but without it
no database test runs on a Windows dev box.
"""

from __future__ import annotations

from triage.db.engine import configure_event_loop_policy

configure_event_loop_policy()
