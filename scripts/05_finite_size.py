"""Day 5: finite-size scaling of the network's estimate of T_c.

For L in (8, 16, 24, 32): generate data (scripts/02), train the Z2-invariant
model, and estimate T_c(L) from the 0.5-crossing.

Finite-size scaling predicts

    T_c(L) - T_c  ~  a * L^(-1/nu)

with nu = 1 for the 2D Ising model. Plot T_c(L) against 1/L: the points
should lie roughly on a straight line, and extrapolating to 1/L -> 0 gives an
estimate of T_c in the thermodynamic limit. Fit the line with
np.polyfit and report the extrapolated value with the exact T_c.

Also plot P(ordered) against the rescaled variable (T - T_c) * L^(1/nu) for
all L: if the network has learned the right physics, the curves collapse.
That "data collapse" plot is the highlight of the README.

Stretch: since the models work for any L, also try training on L = 16 and
evaluating on L = 32 without retraining.
"""


def main():
    raise NotImplementedError("Day 5")


if __name__ == "__main__":
    main()
