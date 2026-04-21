import random
import sys
import time
from dataclasses import dataclass
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


def remove_redundant_columns(sol: Solution, instance: Instance) -> None:
    """Remove redundant columns from the solution."""
    order = sorted(sol.columns, key=lambda j: (instance.costs[j], len(instance.col_rows[j])), reverse=True)
    for j in order:
        if not sol.selected[j]:
            continue
        if all(sol.cover_count[i] >= 2 for i in instance.col_rows[j]):
            remove_column(sol, instance, j)


def greedy_initial_solution(
    instance: Instance,
    rng: random.Random,
    rcl_factor: float = 0.08,
) -> Solution:
    """
    Construct an initial solution with a weighted greedy rule.

    Each uncovered row gets weight 1 / frequency(row), so columns covering rare
    rows receive more credit. Among the best-scoring columns we use a small
    restricted candidate list to avoid being fully deterministic.
    """
    sol = make_empty_solution(instance)
    uncovered = set(range(instance.m))
    # weights = 1 / frequency of the row
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

            # tie-break in favor of columns covering more new rows.
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

    remove_redundant_columns(sol, instance)
    sol.columns.sort()
    return sol


def write_solution(path: Path, sol: Solution) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{sol.cost}\n")
        f.write(" ".join(map(str, sol.columns)))
        f.write("\n")


def log_solution(sol: Solution, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"#### Feasible solution of value {sol.cost} [time {elapsed:.3f}]")


def build_initial_solution(instance: Instance, seed: int = 0) -> Solution:
    rng = random.Random(seed)
    sol = greedy_initial_solution(instance, rng=rng)
    if not sol.is_feasible():
        raise RuntimeError("Internal error: greedy produced an infeasible solution")
    return sol


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} INSTANCE [SEED]", file=sys.stderr)
        return 1

    instance_path = Path(argv[1])
    seed = int(argv[2]) if len(argv) >= 3 else 0

    start_time = time.time()
    with instance_path.open("r", encoding="utf-8") as f:
        instance = read_instance(f)

    sol = build_initial_solution(instance, seed=seed)
    log_solution(sol, start_time)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    write_solution(results_dir / f"{instance_path.name}.1.sol", sol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
