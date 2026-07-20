from __future__ import annotations

from rs.core.contracts.checks import CheckCode


MISSING_FIELD = CheckCode("missing_field")
INVALID_STATUS = CheckCode("invalid_status")
ELIGIBILITY_REJECTED = CheckCode("eligibility_rejected")

__all__ = ["ELIGIBILITY_REJECTED", "INVALID_STATUS", "MISSING_FIELD"]
