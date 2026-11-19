"""Split used_car_sales.csv into a training set and a held-out test set.

Fit every model on the training set (70% of the cars). The test set (30%)
is only for evaluating how well that fit generalizes.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "used_car_sales.csv"
TRAIN_PATH = Path(__file__).resolve().parent / "training_set.csv"
TEST_PATH = Path(__file__).resolve().parent / "test_set.csv"

TEST_FRACTION = 0.30
RANDOM_STATE = 42


def main() -> None:
    sales = pd.read_csv(DATA_PATH)
    n = len(sales)
    n_test = round(n * TEST_FRACTION)

    test = sales.sample(n=n_test, random_state=RANDOM_STATE)
    train = sales.drop(index=test.index).sort_index()
    test = test.sort_index()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Loaded {n} cars from {DATA_PATH.name}")
    print(f"Training set: {len(train)} cars ({len(train) / n:.0%}) -> {TRAIN_PATH.name}")
    print(f"Test set:     {len(test)} cars ({len(test) / n:.0%}) -> {TEST_PATH.name}")
    print("Fit models on the training set only. Use the test set only to evaluate fit.")


if __name__ == "__main__":
    main()
