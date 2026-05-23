from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class IncumbentEvent:
    time: float
    cost: int


def load_trace(path: Path) -> List[IncumbentEvent]:
    events: List[IncumbentEvent] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() == "time,cost":
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid trace line: {line!r}")

            events.append(IncumbentEvent(time=float(parts[0]), cost=int(parts[1])))

    events.sort(key=lambda event: event.time)
    return events


def primal_gap(cost: int, best_known: int) -> float:
    if cost <= best_known:
        return 0.0
    return (cost - best_known) / cost


def primal_integral(events: Sequence[IncumbentEvent], best_known: int, time_limit: float) -> float:
    if time_limit <= 0.0:
        return 0.0

    total = 0.0
    last_time = 0.0
    current_gap = 1.0

    for event in events:
        event_time = max(0.0, min(event.time, time_limit))
        if event_time < last_time:
            continue

        total += current_gap * (event_time - last_time)
        current_gap = primal_gap(event.cost, best_known)
        last_time = event_time

        if last_time >= time_limit:
            return total

    total += current_gap * (time_limit - last_time)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the primal integral and normalized score from an incumbent trace."
    )
    parser.add_argument("trace", type=Path, help="Path to the CSV trace file written by the solver")
    parser.add_argument("best_known", type=int, help="Best known objective value for the instance")
    parser.add_argument(
        "time_limit",
        type=float,
        nargs="?",
        default=600.0,
        help="Time limit T used for the score (default: 600)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    events = load_trace(args.trace)
    integral = primal_integral(events, args.best_known, args.time_limit)
    score = 1.0 - (integral / args.time_limit if args.time_limit > 0.0 else 0.0)

    print(f"Trace file: {args.trace}")
    print(f"Best known: {args.best_known}")
    print(f"Time limit: {args.time_limit:.6f}")
    print(f"Primal integral: {integral:.6f}")
    print(f"Score: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
