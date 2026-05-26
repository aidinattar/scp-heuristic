from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.sc_solver import (
	add_column,
	IncumbentTracker,
	Instance,
	Solution,
	copy_solution,
	get_uncovered_rows,
	one_drop_local_search,
	read_instance,
	remove_column,
	remove_redundant_columns,
	write_solution,
	build_initial_solutions,
)
from utils.solution_checker import checker, readInstance as checker_readInstance


def select_greedy_column(instance: Instance, sol: Solution, candidate_cols: set[int] | None = None) -> int:
	best_col = -1
	best_score = -1.0
	best_new_rows = 0

	columns = range(instance.n) if candidate_cols is None else candidate_cols

	for j in columns:
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

	return best_col


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

    # Mostly remove suspicious columns.
    while len(to_remove) < n_remove and elite_pool:
        to_remove.add(rng.choice(elite_pool))

    # Fill with random selected columns if needed.
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
	deadline = time.time() + time_limit

	# ILS / SA parameters
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
	while time.time() < deadline:
		iteration += 1

		candidate = smart_perturb_solution(
			current,
			instance,
			rng=rng,
			strength=perturbation_strength,
			repair_rcl_factor=repair_rcl_factor,
		)
		if candidate is None:
			temperature = max(min_temperature, temperature * cooling)
			continue

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
			remove_redundant_columns(candidate, instance)
			best = copy_solution(candidate)
			tracker.update(best)

		temperature = max(min_temperature, temperature * cooling)

		if iteration % restart_every == 0 and current.cost > best.cost:
			current = copy_solution(best)

	return best


def validate_with_solution_checker(instance_path: Path, sol: Solution) -> None:
	with instance_path.open("r", encoding="utf-8") as fp:
		objective, matrix = checker_readInstance(fp)
	checker(objective, matrix, sol.cost, sol.columns)


def main(argv: Sequence[str]) -> int:
	if len(argv) < 2:
		print(f"Usage: {argv[0]} INSTANCE [SEED] [TIME_LIMIT]", file=sys.stderr)
		return 1

	instance_path = Path(argv[1])
	seed = int(argv[2]) if len(argv) >= 3 else 0
	time_limit = float(argv[3]) if len(argv) >= 4 else 30.0

	start_time = time.time()
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
		validate_with_solution_checker(instance_path, best)
	finally:
		tracker.close()
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
