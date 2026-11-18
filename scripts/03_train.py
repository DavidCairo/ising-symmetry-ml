"""Day 3: train a classifier and locate T_c from its output.

Usage idea:
  python scripts/03_train.py --model plain --L 16 --epochs 10 --seed 0

Steps:
  1. load data/train_L{L}.npz, split off ~10% as validation set
  2. build the model, Adam optimizer, BCEWithLogitsLoss
  3. train for a few epochs, printing train loss and val accuracy per epoch
  4. on the test set: average predicted P(ordered) per temperature
  5. estimate T_c as the temperature where that curve crosses 0.5
     (linear interpolation between the two neighbouring points is fine)
  6. save the model weights to results/ and the figure
     figures/prediction_vs_T_{model}_L{L}.png (with the exact T_c marked)

Write the estimated T_c and the relative error w.r.t. Onsager into the README.
"""


def main():
    raise NotImplementedError("Day 3")


if __name__ == "__main__":
    main()
