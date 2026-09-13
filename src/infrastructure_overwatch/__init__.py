"""Infrastructure Overwatch: a defensive, analyst-in-the-loop computer-vision
pipeline for critical-infrastructure corridor monitoring.

Scope, stated up front: this package detects, tracks, and routes alerts about
sensor observations to a human analyst. It contains no targeting, engagement,
or fires logic, and nothing in this codebase should be extended into one. See
docs/USER_MANUAL.md and docs/METHODOLOGY_AND_LIMITATIONS.md before trusting
any output.
"""

from __future__ import annotations

__version__ = "0.1.0"
