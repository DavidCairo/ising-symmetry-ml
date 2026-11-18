# Learning Z2 symmetry breaking in the 2D Ising model

![tests](https://github.com/DavidCairo/ising-symmetry-ml/actions/workflows/tests.yml/badge.svg)

<!-- One-paragraph summary, written LAST (Day 6). What question, what you did, the main result with a number. -->

**Question.** Can a neural network trained only on clearly ordered and clearly
disordered configurations locate the critical temperature of the 2D Ising
model, and does building the model's Z2 symmetry (s → −s) into the
architecture make it learn more efficiently?

## Results

<!-- Day 6: 3-4 figures, each with one or two sentences of interpretation. -->

1. **Validating the sampler.** <!-- figures/magnetization_vs_T.png -->
2. **The network finds T_c.** <!-- prediction vs T, estimated T_c vs Onsager 2.269 -->
3. **Symmetry helps.** <!-- data-efficiency plot + the m>0-only experiment -->
4. **Finite-size scaling.** <!-- T_c(L) vs 1/L and the data collapse -->

## Method

<!-- Short: Metropolis with checkerboard updates; labels from T < T_c;
     train/test temperature gap around T_c; model architectures; how the
     Z2 invariance is enforced and why it is exact. -->

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # all tests should pass
python scripts/01_validate_mc.py
python scripts/02_generate_data.py
python scripts/03_train.py --model plain
python scripts/04_compare_symmetry.py
python scripts/05_finite_size.py
```

<!-- Add approximate run times on your machine. -->

## Repository layout

```
src/ising_ml/   lattice observables, exact results, Monte Carlo, data, models, training
scripts/        one script per experiment; each writes its figure to figures/
tests/          unit tests, incl. an exact-enumeration check of the sampler
figures/        figures used in this README
```

## References

- J. Carrasquilla and R. G. Melko, *Machine learning phases of matter*, Nature Physics 13, 431 (2017).
- L. Onsager, *Crystal statistics. I.*, Phys. Rev. 65, 117 (1944); C. N. Yang, Phys. Rev. 85, 808 (1952).
