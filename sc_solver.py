#!/usr/bin/env python3

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, TextIO

@dataclass(frozen=True)
class Instance:
    m: int
    n: int
    costs: List[int]
    col_rows: List[List[int]]
    row_cols: List[List[int]]


@dataclass
class Solution:
    selected: List[bool]
    columns: List[int]
    cost: int
    cover_count: List[int]

    def is_feasible(self) -> bool:
        return all(cnt > 0 for cnt in self.cover_count)


@dataclass
class IncumbentTracker:
    instance_name: str
    results_dir: Path
    start_time: float
    best_cost: int | None = None
    counter: int = 0
    trace_path: Path | None = None
    trace_file: TextIO | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace_path is None:
            self.trace_path = self.results_dir / f"{self.instance_name}.trace.csv"

        self.trace_file = self.trace_path.open("w", encoding="utf-8")
        self.trace_file.write("time,cost\n")
        self.trace_file.flush()

    def close(self) -> None:
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None

    def log_incumbent(self, elapsed: float, cost: int) -> None:
        if self.trace_file is None:
            return

        self.trace_file.write(f"{elapsed:.6f},{cost}\n")
        self.trace_file.flush()

    def update(self, sol: Solution) -> None:
        if not sol.is_feasible():
            return
        if self.best_cost is not None and sol.cost >= self.best_cost:
            return

        self.best_cost = sol.cost
        self.counter += 1

        sol.columns.sort()
        elapsed = time.monotonic() - self.start_time

        print(f"#### Feasible solution of value {sol.cost} [time {elapsed:.3f}]", flush=True)
        write_solution(self.results_dir / f"{self.instance_name}.{self.counter}.sol", sol)
        self.log_incumbent(elapsed, sol.cost)


def read_instance(fp: TextIO) -> Instance:
    first = fp.readline().split()
    if len(first) != 2:
        raise ValueError("Invalid header: expected 'm n'")

    m, n = map(int, first)
    if m <= 0 or n <= 0:
        raise ValueError("Invalid instance dimensions")

    costs: List[int] = [0] * n
    col_rows: List[List[int]] = [[] for _ in range(n)]
    row_cols: List[List[int]] = [[] for _ in range(m)]

    for j in range(n):
        parts = fp.readline().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid column line for column {j}")

        cost = int(parts[0])
        count = int(parts[1])
        rows = [int(x) - 1 for x in parts[2:]]

        if count != len(rows):
            raise ValueError(f"Column {j}: expected {count} covered rows, got {len(rows)}")
        if cost <= 0:
            raise ValueError(f"Column {j}: cost must be positive")

        costs[j] = cost
        col_rows[j] = rows

        for i in rows:
            if i < 0 or i >= m:
                raise ValueError(f"Column {j}: row index out of range")
            row_cols[i].append(j)

    for i, cols in enumerate(row_cols):
        if not cols:
            raise ValueError(f"Row {i} cannot be covered by any column")

    return Instance(m=m, n=n, costs=costs, col_rows=col_rows, row_cols=row_cols)


def add_column(sol: Solution, instance: Instance, j: int) -> None:
    if sol.selected[j]:
        return

    sol.selected[j] = True
    sol.columns.append(j)
    sol.cost += instance.costs[j]
    for i in instance.col_rows[j]:
        sol.cover_count[i] += 1


def remove_column(sol: Solution, instance: Instance, j: int) -> None:
    if not sol.selected[j]:
        return

    sol.selected[j] = False
    sol.columns.remove(j)
    sol.cost -= instance.costs[j]
    for i in instance.col_rows[j]:
        sol.cover_count[i] -= 1


def make_empty_solution(instance: Instance) -> Solution:
    return Solution(
        selected=[False] * instance.n,
        columns=[],
        cost=0,
        cover_count=[0] * instance.m,
    )


def copy_solution(sol: Solution) -> Solution:
    return Solution(
        selected=sol.selected.copy(),
        columns=sol.columns.copy(),
        cost=sol.cost,
        cover_count=sol.cover_count.copy(),
    )


def get_uncovered_rows(sol: Solution) -> set[int]:
    return {i for i, cnt in enumerate(sol.cover_count) if cnt == 0}


def remove_redundant_columns(sol: Solution, instance: Instance) -> None:
    """Remove redundant columns from the solution."""
    order = sorted(sol.columns, key=lambda j: (instance.costs[j], len(instance.col_rows[j])), reverse=True)
    for j in order:
        if not sol.selected[j]:
            continue
        if all(sol.cover_count[i] >= 2 for i in instance.col_rows[j]):
            remove_column(sol, instance, j)


def all_columns_solution(
    instance: Instance,
    tracker: IncumbentTracker | None = None,
) -> Solution:
    """Return the trivial feasible solution selecting all columns."""
    sol = make_empty_solution(instance)
    for j in range(instance.n):
        add_column(sol, instance, j)

    if tracker is not None:
        tracker.update(sol)

    return sol


def fast_greedy_solution(
    instance: Instance,
    tracker: IncumbentTracker | None = None,
) -> Solution:
    """Build a quick first solution using plain uncovered rows per unit cost."""
    sol = make_empty_solution(instance)
    uncovered = set(range(instance.m))

    while uncovered:
        best_col = -1
        best_score = -1.0
        best_new_rows = 0

        for j in range(instance.n):
            if sol.selected[j]:
                continue

            new_rows = 0
            for i in instance.col_rows[j]:
                if sol.cover_count[i] == 0:
                    new_rows += 1

            if new_rows == 0:
                continue

            score = new_rows / instance.costs[j]
            if score > best_score or (score == best_score and new_rows > best_new_rows):
                best_score = score
                best_col = j
                best_new_rows = new_rows

        if best_col < 0:
            raise RuntimeError("No candidate column found while uncovered rows remain")

        add_column(sol, instance, best_col)
        for i in instance.col_rows[best_col]:
            uncovered.discard(i)

    if tracker is not None:
        tracker.update(sol)

    remove_redundant_columns(sol, instance)

    if tracker is not None:
        tracker.update(sol)

    sol.columns.sort()
    return sol


def weighted_greedy_solution(
    instance: Instance,
    rng: random.Random,
    tracker: IncumbentTracker | None = None,
    rcl_factor: float = 0.08,
) -> Solution:
    """
    Construct a solution with a weighted greedy rule.

    Each uncovered row gets weight 1 / frequency(row), so columns covering rare
    rows receive more credit. Among the best-scoring columns we use a small
    restricted candidate list to avoid being fully deterministic.
    """
    sol = make_empty_solution(instance)
    uncovered = set(range(instance.m))
    row_weight = [1.0 / len(instance.row_cols[i]) for i in range(instance.m)]

    while uncovered:
        scores: List[tuple[float, int]] = []
        best_score = -1.0

        for j in range(instance.n):
            if sol.selected[j]:
                continue

            gain = 0.0
            new_rows = 0
            for i in instance.col_rows[j]:
                if sol.cover_count[i] == 0:
                    gain += row_weight[i]
                    new_rows += 1

            if new_rows == 0:
                continue

            score = gain / instance.costs[j]
            score *= 1.0 + 1e-6 * new_rows
            scores.append((score, j))
            if score > best_score:
                best_score = score

        if not scores:
            raise RuntimeError("No candidate column found while uncovered rows remain")

        threshold = best_score * (1.0 - rcl_factor)
        rcl = [j for score, j in scores if score >= threshold]
        chosen = rng.choice(rcl)
        add_column(sol, instance, chosen)

        for i in instance.col_rows[chosen]:
            uncovered.discard(i)

    if tracker is not None:
        tracker.update(sol)

    remove_redundant_columns(sol, instance)

    if tracker is not None:
        tracker.update(sol)

    sol.columns.sort()
    return sol


def repair_solution(sol: Solution, instance: Instance, uncovered: set[int]) -> bool:
    """Repair uncovered rows with a simple greedy rule."""
    while uncovered:
        candidate_cols: set[int] = set()
        for i in uncovered:
            candidate_cols.update(instance.row_cols[i])

        best_col = -1
        best_score = -1.0
        best_new_rows = 0

        for j in candidate_cols:
            if sol.selected[j]:
                continue

            new_rows = 0
            for i in instance.col_rows[j]:
                if i in uncovered:
                    new_rows += 1

            if new_rows == 0:
                continue

            score = new_rows / instance.costs[j]
            if score > best_score or (score == best_score and new_rows > best_new_rows):
                best_score = score
                best_col = j
                best_new_rows = new_rows

        if best_col < 0:
            return False

        add_column(sol, instance, best_col)
        for i in instance.col_rows[best_col]:
            uncovered.discard(i)

    return True


def one_drop_local_search(
    sol: Solution,
    instance: Instance,
    tracker: IncumbentTracker,
    max_passes: int = 3,
) -> Solution:
    """Try to improve a solution by removing one column and repairing."""
    current = copy_solution(sol)

    for _ in range(max_passes):
        improved = False
        order = sorted(current.columns, key=lambda j: instance.costs[j], reverse=True)

        for j in order:
            if not current.selected[j]:
                continue

            candidate = copy_solution(current)
            remove_column(candidate, instance, j)

            uncovered = get_uncovered_rows(candidate)
            if uncovered and not repair_solution(candidate, instance, uncovered):
                continue

            remove_redundant_columns(candidate, instance)
            candidate.columns.sort()

            if candidate.is_feasible() and candidate.cost < current.cost:
                current = candidate
                tracker.update(current)
                improved = True
                break

        if not improved:
            break

    return current


def row_based_greedy_solution(
    instance: Instance,
    tracker: IncumbentTracker | None = None,
) -> Solution:
    """Build a quick solution by selecting columns from uncovered critical rows."""
    sol = make_empty_solution(instance)
    uncovered = set(range(instance.m))

    while uncovered:
        row = min(uncovered, key=lambda i: len(instance.row_cols[i]))

        best_col = -1
        best_score = -1.0
        best_new_rows = 0

        for j in instance.row_cols[row]:
            if sol.selected[j]:
                continue

            new_rows = 0
            for i in instance.col_rows[j]:
                if i in uncovered:
                    new_rows += 1

            if new_rows == 0:
                continue

            score = new_rows / instance.costs[j]

            if score > best_score or (score == best_score and new_rows > best_new_rows):
                best_score = score
                best_col = j
                best_new_rows = new_rows

        if best_col < 0:
            raise RuntimeError("No candidate column found while uncovered rows remain")

        add_column(sol, instance, best_col)

        for i in instance.col_rows[best_col]:
            uncovered.discard(i)

    if tracker is not None:
        tracker.update(sol)

    remove_redundant_columns(sol, instance)

    if tracker is not None:
        tracker.update(sol)

    sol.columns.sort()
    return sol


def randomized_row_based_greedy_solution(
    instance: Instance,
    rng: random.Random,
    tracker: IncumbentTracker | None = None,
    row_rcl_size: int = 20,
    col_rcl_factor: float = 0.10,
) -> Solution:
    """Build a solution by selecting hard uncovered rows and randomized good columns."""
    sol = make_empty_solution(instance)
    uncovered = set(range(instance.m))

    while uncovered:
        hard_rows = sorted(uncovered, key=lambda i: len(instance.row_cols[i]))[:row_rcl_size]
        row = rng.choice(hard_rows)

        scored_cols: list[tuple[float, int, int]] = []
        best_score = -1.0

        for j in instance.row_cols[row]:
            if sol.selected[j]:
                continue

            new_rows = 0
            for i in instance.col_rows[j]:
                if i in uncovered:
                    new_rows += 1

            if new_rows == 0:
                continue

            score = new_rows / instance.costs[j]
            scored_cols.append((score, new_rows, j))

            if score > best_score:
                best_score = score

        if not scored_cols:
            raise RuntimeError("No candidate column found while uncovered rows remain")

        threshold = best_score * (1.0 - col_rcl_factor)
        rcl = [(score, new_rows, j) for score, new_rows, j in scored_cols if score >= threshold]

        _, _, chosen = rng.choice(rcl)
        add_column(sol, instance, chosen)

        for i in instance.col_rows[chosen]:
            uncovered.discard(i)

    if tracker is not None:
        tracker.update(sol)

    remove_redundant_columns(sol, instance)

    if tracker is not None:
        tracker.update(sol)

    sol.columns.sort()
    return sol


def write_solution(path: Path, sol: Solution) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{sol.cost}\n")
        f.write(" ".join(map(str, sol.columns)))
        f.write("\n")


def build_initial_solutions(
    instance: Instance,
    tracker: IncumbentTracker,
    seed: int = 0,
    n_restarts: int = 10,
) -> Solution:
    rng = random.Random(seed)

    trivial_sol = all_columns_solution(instance, tracker=tracker)
    candidates = [trivial_sol]

    for _ in range(n_restarts):
        sol = randomized_row_based_greedy_solution(
            instance,
            rng=rng,
            tracker=tracker,
            row_rcl_size=20,
            col_rcl_factor=0.10,
        )
        candidates.append(sol)

    for sol in candidates:
        if not sol.is_feasible():
            raise RuntimeError("Internal error: construction produced an infeasible solution")

    best = min(candidates, key=lambda sol: sol.cost)
    best = one_drop_local_search(best, instance, tracker=tracker, max_passes=3)
    return best


def randomized_repair_solution(
    sol: Solution,
    instance: Instance,
    uncovered: set[int],
    rng: random.Random,
    rcl_factor: float = 0.10,
) -> bool:
    """Repair uncovered rows using a randomized greedy rule."""
    while uncovered:
        candidate_cols: set[int] = set()
        for i in uncovered:
            candidate_cols.update(instance.row_cols[i])

        scored_cols: list[tuple[float, int, int]] = []
        best_score = -1.0

        for j in candidate_cols:
            if sol.selected[j]:
                continue

            new_rows = 0
            for i in instance.col_rows[j]:
                if i in uncovered:
                    new_rows += 1

            if new_rows == 0:
                continue

            score = new_rows / instance.costs[j]
            scored_cols.append((score, new_rows, j))

            if score > best_score:
                best_score = score

        if not scored_cols:
            return False

        threshold = best_score * (1.0 - rcl_factor)
        rcl = [(score, new_rows, j) for score, new_rows, j in scored_cols if score >= threshold]

        _, _, chosen = rng.choice(rcl)
        add_column(sol, instance, chosen)

        for i in instance.col_rows[chosen]:
            uncovered.discard(i)

    return True


def private_row_count(sol: Solution, instance: Instance, j: int) -> int:
    """Rows that would become uncovered if column j were removed."""
    count = 0
    for i in instance.col_rows[j]:
        if sol.cover_count[i] == 1:
            count += 1
    return count


def column_badness(sol: Solution, instance: Instance, j: int) -> float:
    """High value means: expensive and not very critical."""
    private = private_row_count(sol, instance, j)
    return instance.costs[j] / (1.0 + private)


def smart_perturb_solution(
    sol: Solution,
    instance: Instance,
    rng: random.Random,
    strength: float = 0.08,
    elite_fraction: float = 0.50,
    repair_rcl_factor: float = 0.10,
) -> Solution:
    """
    Perturb a solution by removing a mix of bad columns and random columns,
    then repair it.
    """
    candidate = copy_solution(sol)

    if not candidate.columns:
        return candidate

    n_remove = max(1, int(strength * len(candidate.columns)))
    n_remove = min(n_remove, len(candidate.columns))

    ranked = sorted(
        candidate.columns,
        key=lambda j: column_badness(candidate, instance, j),
        reverse=True,
    )

    elite_size = max(1, int(elite_fraction * len(ranked)))
    elite_pool = ranked[:elite_size]

    to_remove: set[int] = set()

    while len(to_remove) < n_remove and elite_pool:
        to_remove.add(rng.choice(elite_pool))

    while len(to_remove) < n_remove:
        to_remove.add(rng.choice(candidate.columns))

    for j in to_remove:
        if candidate.selected[j]:
            remove_column(candidate, instance, j)

    uncovered = get_uncovered_rows(candidate)
    if uncovered:
        ok = randomized_repair_solution(
            candidate,
            instance,
            uncovered,
            rng=rng,
            rcl_factor=repair_rcl_factor,
        )
        if not ok:
            return copy_solution(sol)

    remove_redundant_columns(candidate, instance)
    candidate.columns.sort()
    return candidate


def accept_candidate(current_cost: int, candidate_cost: int, temperature: float, rng: random.Random) -> bool:
    """Simulated annealing acceptance: always accept improvements, sometimes accept worse moves."""
    if candidate_cost <= current_cost:
        return True
    if temperature <= 0.0:
        return False
    return rng.random() < math.exp(-(candidate_cost - current_cost) / temperature)


def iterated_local_search_sa(
    instance: Instance,
    tracker: IncumbentTracker,
    seed: int = 0,
    time_limit: float = 30.0,
) -> Solution:
    """Iterated local search with greedy local search, cleanup, perturbation, and SA acceptance."""
    rng = random.Random(seed)
    deadline = time.monotonic() + time_limit

    n_restarts = 5
    perturbation_strength = 0.06
    repair_rcl_factor = 0.10
    cooling = 0.995
    min_temperature = 1e-3
    restart_every = 100

    current = build_initial_solutions(instance, tracker, seed, n_restarts)
    best = copy_solution(current)
    temperature = max(1.0, float(best.cost) * 0.05)

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1

        candidate = smart_perturb_solution(
            current,
            instance,
            rng=rng,
            strength=perturbation_strength,
            repair_rcl_factor=repair_rcl_factor,
        )

        remove_redundant_columns(candidate, instance)
        candidate.columns.sort()

        candidate = one_drop_local_search(candidate, instance, tracker=tracker, max_passes=2)
        remove_redundant_columns(candidate, instance)
        candidate.columns.sort()
        if not candidate.is_feasible():
            temperature = max(min_temperature, temperature * cooling)
            continue

        if accept_candidate(current.cost, candidate.cost, temperature, rng):
            current = candidate

        if candidate.cost < best.cost:
            best = copy_solution(candidate)
            tracker.update(best)

        temperature = max(min_temperature, temperature * cooling)

        if iteration % restart_every == 0 and current.cost > best.cost:
            current = copy_solution(best)

    return best


def validate_solution(instance: Instance, sol: Solution) -> None:
    if not sol.is_feasible():
        raise RuntimeError("Internal error: final solution is infeasible")

    objective = sum(instance.costs[j] for j in sol.columns)
    if objective != sol.cost:
        raise RuntimeError(
            f"Internal error: stored cost {sol.cost} does not match recomputed cost {objective}"
        )


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} INSTANCE [SEED] [TIME_LIMIT]", file=sys.stderr)
        return 1

    instance_path = Path(argv[1])
    seed = int(argv[2]) if len(argv) >= 3 else 0
    time_limit = float(argv[3]) if len(argv) >= 4 else 600.0

    start_wall_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    start_time = time.monotonic()

    print(f"Start time: {start_wall_time}", flush=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with instance_path.open("r", encoding="utf-8") as f:
        instance = read_instance(f)

    tracker = IncumbentTracker(
        instance_name=instance_path.name,
        results_dir=results_dir,
        start_time=start_time,
    )

    try:
        best = iterated_local_search_sa(instance, tracker=tracker, seed=seed, time_limit=time_limit)
        write_solution(results_dir / f"{instance_path.name}.final.sol", best)
        validate_solution(instance, best)
    finally:
        tracker.close()
        end_wall_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        total_elapsed = time.monotonic() - start_time
        print(f"End time: {end_wall_time}", flush=True)
        print(f"Total elapsed: {total_elapsed:.3f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
