"""Day 4: does building in the Z2 symmetry help?

Experiment 1 -- data efficiency:
  Train PlainCNN and Z2InvariantCNN on training sets of increasing size
  (e.g. 50, 100, 200, 500, 1000, all configurations). For each size and model,
  repeat with 3-5 seeds and record the test accuracy and the T_c estimate.
  Plot accuracy vs. training-set size (log x-axis), mean +- std over seeds.

Experiment 2 -- the symmetry test a physicist would do:
  Train PlainCNN ONLY on configurations with m > 0 (throw away the
  ordered configurations with negative magnetization). Then evaluate on
  ordered configurations with m < 0. What happens to the plain network?
  And to the Z2-invariant one? This is the cleanest demonstration of why
  symmetry matters: the plain network has never seen the "other" vacuum.

Bonus: compare parameter counts (ising_ml.models.count_parameters).

Save figures to figures/ and summarise the result in the README in 3-4 sentences.
"""


def main():
    raise NotImplementedError("Day 4")


if __name__ == "__main__":
    main()
