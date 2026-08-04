from __future__ import annotations

import argparse
from pathlib import Path

from rs_sim.scheduler.prediction.timing import (
    build_rank_timing_profile,
    save_rank_timing_profile,
)
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model/layer/rank timing profile from calibration-only trace "
            "fixtures. Evaluated fixtures must not be listed here."
        )
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixtures = []
    for path in args.fixture:
        fixture = load_fixture(path)
        validate_fixture(fixture)
        fixtures.append(fixture)
    profile = build_rank_timing_profile(fixtures, profile_id=args.profile_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rank_timing_profile(profile, args.output)
    print(args.output)
    print(profile.profile_digest)


if __name__ == "__main__":
    main()
