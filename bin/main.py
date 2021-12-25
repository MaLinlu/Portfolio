from datetime import datetime
import os
from typing import Tuple, List
from pandas.core.frame import DataFrame
from pytz import UTC

from src.FTX.client import FtxClient
from src.EfficientPortfolio.portfilio import EfficientPortfilio

RESOLUTION = 3600


def get_expected_return(
    client: FtxClient, markets: List, start_time: datetime, end_time: datetime
) -> Tuple[DataFrame, DataFrame]:
    """get mean and hourly historical data"""
    returns_hourly = DataFrame()
    start_time_float = start_time.timestamp()
    end_time_float = end_time.timestamp()

    for market in markets:
        historical_data = client.get_historical_data(
            market=market,
            resolution=RESOLUTION,
            start_time=start_time_float,
            end_time=end_time_float,
        )
        historical_data_df = DataFrame(historical_data)
        returns_hourly[f"{market}"] = historical_data_df["close"].pct_change().dropna()
    expected_returns_avg = returns_hourly.mean(axis=0)

    return expected_returns_avg, returns_hourly


def _main():

    API_KEY = os.environ.get("FTX_API_KEY")
    API_SECRET = os.environ.get("FTX_API_SECRET")
    client = FtxClient(API_KEY, API_SECRET)

    # datetime(year, month, day, hour, minute, second, microsecond)
    start_time = datetime(2021, 10, 1, 0, tzinfo=UTC)
    end_time = datetime(2021, 10, 1, 23, tzinfo=UTC)

    markets = ["BTC-PERP", "ETH-PERP", "ADA-PERP"]

    # returns
    expected_returns, returns_hourly = get_expected_return(
        client, markets, start_time, end_time
    )

    # covariance
    returns_cov = returns_hourly.cov()

    efficient_port = EfficientPortfilio(expected_returns, returns_cov)
    efficient_port.display_ef_with_selected()


if __name__ == "__main__":
    _main()
