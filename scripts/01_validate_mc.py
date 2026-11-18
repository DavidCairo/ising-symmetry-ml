"""Day 2: validate the Monte Carlo sampler against Onsager's exact solution.

Produces figures/magnetization_vs_T.png:
  <|m|> versus T for L = 8, 16, 32, with the exact infinite-volume curve
  (ising_ml.exact.spontaneous_magnetization) on top and T_c marked.

What you should see: finite lattices follow the exact curve far from T_c and
get rounded off near T_c, more so for smaller L (finite-size effects).
Write one or two sentences about this in the README.

Outline:
  1. temperatures = np.linspace(1.0, 3.5, ~25)
  2. for each L and T: samples = simulate(...); record mean |m| and its
     standard error
  3. plot with error bars, add the exact curve and a vertical line at T_c
  4. save to figures/ (create the folder if needed) -- dpi=150 is enough

Tip: use argparse so that e.g. `python scripts/01_validate_mc.py --quick`
runs a fast low-statistics version. Print how long the run took.
"""


def main():
    raise NotImplementedError("Day 2")


if __name__ == "__main__":
    main()
