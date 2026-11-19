"""Fit the sale-price model on the training set only."""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

TRAIN_PATH = Path(__file__).resolve().parent / "training_set.csv"
TEST_PATH = Path(__file__).resolve().parent / "test_set.csv"
AUCTION_PATH = Path(__file__).resolve().parent / "auction_lots.csv"
COST = 1500

FULL_FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident + Num_Owners"
REDUCED_FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident"


def test_r_squared(actual, predicted):
    residual_sum_of_squares = ((actual - predicted) ** 2).sum()
    total_sum_of_squares = ((actual - actual.mean()) ** 2).sum()
    return 1 - residual_sum_of_squares / total_sum_of_squares


def test_rmse(actual, predicted):
    return (((actual - predicted) ** 2).mean()) ** 0.5


def baseline_decision(auction_price, mean_sale_price):
    if auction_price + COST < mean_sale_price:
        return "buy"
    return "pass"


def v1_decision(features, auction_price, model):
    predicted_sale_price = float(model.predict(features).iloc[0])
    if auction_price + COST < predicted_sale_price:
        return "buy"
    return "pass"


def rule_results(lots, decisions):
    profits = []
    for decision, (_, car) in zip(decisions, lots.iterrows()):
        if decision == "buy":
            profits.append(car["Later_Sale_Price"] - car["Auction_Price"] - COST)
    cars_bought = len(profits)
    cars_that_lose_money = sum(profit < 0 for profit in profits)
    total_profit = sum(profits)
    return cars_bought, cars_that_lose_money, total_profit


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    full = smf.ols(FULL_FORMULA, data=train).fit()
    reduced = smf.ols(REDUCED_FORMULA, data=train).fit()

    print(full.summary())
    print()

    for name, coef in full.params.items():
        print(f"{name} coefficient: ${round(coef):,}")

    for name, pvalue in full.pvalues.items():
        print(f"{name} p-value: {pvalue:.4f}")

    print(f"R-squared: {full.rsquared:.4f}")
    print(f"Adjusted R-squared: {full.rsquared_adj:.4f}")

    print()
    print("Reduced model")
    print(reduced.summary())
    print()

    print(f"Full model adjusted R-squared: {full.rsquared_adj:.4f}")
    print(f"Full model R-squared: {full.rsquared:.4f}")
    print(f"Reduced model adjusted R-squared: {reduced.rsquared_adj:.4f}")
    print(f"Reduced model R-squared: {reduced.rsquared:.4f}")

    test = pd.read_csv(TEST_PATH)
    actual = test["Sale_Price"]
    baseline = train["Sale_Price"].mean()
    v1 = reduced.predict(test)

    print()
    print(f"Baseline Test R-squared: {test_r_squared(actual, baseline):.4f}")
    print(f"Baseline Test RMSE: ${round(test_rmse(actual, baseline)):,}")
    print(f"V1 Test R-squared: {test_r_squared(actual, v1):.4f}")
    print(f"V1 Test RMSE: ${round(test_rmse(actual, v1)):,}")

    lots = pd.read_csv(AUCTION_PATH)
    mean_sale_price = train["Sale_Price"].mean()
    baseline_decisions = []
    v1_decisions = []
    for index in lots.index:
        car = lots.loc[[index]]
        auction_price = float(car["Auction_Price"].iloc[0])
        baseline_decisions.append(baseline_decision(auction_price, mean_sale_price))
        v1_decisions.append(v1_decision(car, auction_price, reduced))

    baseline_bought, baseline_losers, baseline_profit = rule_results(lots, baseline_decisions)
    v1_bought, v1_losers, v1_profit = rule_results(lots, v1_decisions)

    print()
    print(f"Baseline cars bought: {baseline_bought}")
    print(f"Baseline bought cars that lose money: {baseline_losers}")
    print(f"Baseline total profit: ${baseline_profit:,.0f}")
    print(f"V1 cars bought: {v1_bought}")
    print(f"V1 bought cars that lose money: {v1_losers}")
    print(f"V1 total profit: ${v1_profit:,.0f}")


if __name__ == "__main__":
    main()
