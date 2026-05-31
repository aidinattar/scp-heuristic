# Set Covering Solver — Minimal

Files:
- `sc_solver.py` (main solver) and `sc_solver` (launcher)

Run:
```bash
./sc_solver INSTANCE SEED TIME_LIMIT > log.txt 2>&1
# example: ./sc_solver instances/rail507 0 600 > rail507.log 2>&1
```

Output:
- Prints each new best as: `#### Feasible solution of value xxx [time yyy]`
- Writes `results/INSTANCE.k.sol` and `results/INSTANCE.trace.csv`

Seed (short):
- `SEED` is an integer that controls randomness. Use the same seed to reproduce a run; change the seed to get a different randomized run. Default is `0` when omitted.