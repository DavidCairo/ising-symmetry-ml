"""Day 3: generate training and test sets and save them to data/.

Suggested set-up (adjust if it is too slow on your laptop):
  * L = 16 (later also 8, 24, 32 for Day 5)
  * training temperatures: e.g. 20 values in [1.0, 1.8] and 20 in [2.8, 3.6]
    -- deliberately leave a gap around T_c, so the network never sees
    critical configurations during training;
  * test temperatures: a fine grid over the whole range [1.0, 3.6];
  * a few hundred configurations per temperature.

Save with ising_ml.data.save_dataset to e.g. data/train_L16.npz and
data/test_L16.npz. data/ is git-ignored: datasets are regenerated from the
script, not committed. Use a fixed seed and print it.
"""


def main():
    raise NotImplementedError("Day 3")


if __name__ == "__main__":
    main()
