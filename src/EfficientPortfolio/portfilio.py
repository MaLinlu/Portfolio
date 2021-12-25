from typing import List, Tuple, Any
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as sco


class EfficientPortfilio:
    def __init__(self, mean_returns: DataFrame, cov: DataFrame) -> None:
        self.mean_returns = mean_returns
        self.cov_matrix = cov
        self.num_assets = len(self.mean_returns)
        self.annualised_factor = 365 * 24

    def portfolio_annualised_performance(self, weights) -> Tuple[float, float]:
        returns = np.sum(self.mean_returns * weights) * self.annualised_factor
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(
            self.annualised_factor
        )
        return std, returns

    def _neg_sharpe_ratio(self, weights: List) -> float:
        p_var, p_ret = self.portfolio_annualised_performance(weights)
        return -p_ret / p_var

    def max_sharpe_ratio(self) -> Any:
        num_assets = self.num_assets
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(
            self._neg_sharpe_ratio,
            num_assets
            * [
                1.0 / num_assets,
            ],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result

    def _portfolio_volatility(self, weights) -> float:
        return self.portfolio_annualised_performance(weights)[0]

    def min_variance(self) -> Any:
        num_assets = self.num_assets
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))

        result = sco.minimize(
            self._portfolio_volatility,
            num_assets
            * [
                1.0 / num_assets,
            ],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result

    def efficient_return(self, target) -> Any:
        num_assets = self.num_assets

        def portfolio_return(weights):
            return self.portfolio_annualised_performance(weights)[1]

        constraints = (
            {"type": "eq", "fun": lambda x: portfolio_return(x) - target},
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        )
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = sco.minimize(
            self._portfolio_volatility,
            num_assets
            * [
                1.0 / num_assets,
            ],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result

    def efficient_frontier(self, returns_range) -> List:
        efficients = []
        for ret in returns_range:
            efficients.append(self.efficient_return(ret))
        return efficients

    def display_ef_with_selected(self) -> None:
        tickers = self.mean_returns.index
        # calculate max sharpe portfolio
        max_sharpe = self.max_sharpe_ratio()
        sd_max_sharpe, returns_max_sharpe = self.portfolio_annualised_performance(
            max_sharpe["x"]
        )
        max_sharpe_allocation = {
            ticker: round(max_sharpe["x"][i], 2)
            for i, ticker in enumerate(tickers)
        }
        # calculate min risk portfolio
        min_vol = self.min_variance()
        sd_min_vol, returns_min_vol = self.portfolio_annualised_performance(
            min_vol["x"]
        )
        min_vol_allocation = {
            ticker: round(min_vol["x"][i], 2) for i, ticker in enumerate(tickers)
        }

        # display output
        print("-" * 80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(returns_max_sharpe, 2))
        print("Annualised Volatility:", round(sd_max_sharpe, 2))

        print("Allocation: ", max_sharpe_allocation)
        print("-" * 80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(returns_min_vol, 2))
        print("Annualised Volatility:", round(sd_min_vol, 2))

        print("Allocation: ", min_vol_allocation)
        print("-" * 80)

        # plot individual contracts
        annualised_vol = np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(
            self.annualised_factor
        )
        annualised_returns = self.mean_returns * self.annualised_factor
        _, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(annualised_vol, annualised_returns, marker="o", s=200)

        # plot max sharpe and min vol points
        for i, txt in enumerate(tickers):
            ax.annotate(
                txt,
                (annualised_vol[i], annualised_returns[i]),
                xytext=(10, 0),
                textcoords="offset points",
            )
        ax.scatter(
            sd_max_sharpe,
            returns_max_sharpe,
            marker="*",
            color="r",
            s=500,
            label="Maximum Sharpe ratio",
        )
        ax.scatter(
            sd_min_vol,
            returns_min_vol,
            marker="*",
            color="g",
            s=500,
            label="Minimum volatility",
        )

        # generate and plot the frontier
        target = np.linspace(returns_min_vol, max(annualised_returns), 50)
        efficient_portfolios = self.efficient_frontier(target)
        ax.plot(
            [p["fun"] for p in efficient_portfolios],
            target,
            linestyle="-.",
            color="black",
            label="efficient frontier",
        )
        ax.set_title("Portfolio Optimization with Individual Contracts")
        ax.set_xlabel("annualised volatility")
        ax.set_ylabel("annualised returns")
        ax.legend(labelspacing=0.8)

        # plt.show()
        plt.savefig("EfficientFrontier.png")
