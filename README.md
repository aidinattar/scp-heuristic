# Set Covering Heuristic

Python heuristic for the Set Covering Problem on OR-Library rail instances.

## Structure

- `sc_solver.py` main solver
- `solution_checker.py` solution checker
- `ILS.py` iterated local search solver
- `primal_integral.py` primal integral calculator from the incumbent trace
- `instances/` input instances
- `results/` logs and solutions
- `notes.md` short development notes

## Primal Integral

Both solvers now write an incumbent trace at `results/<instance>.trace.csv`.

Run the solver as usual, then compute the primal integral with:

```bash
uv run utils/primal_integral.py results/rail507.trace.csv <best_known> 600
```

Replace `<best_known>` with the best known objective value for the instance.
