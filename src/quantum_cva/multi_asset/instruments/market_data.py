from pathlib import Path
import numpy as np
import pandas as pd


class MarketData:
    def __init__(
        self,
        discount_curve: pd.DataFrame,
        credit_spreads: pd.DataFrame,
        recovery_rate: float,
        prices: pd.DataFrame,
        log_returns: pd.DataFrame,
        atm_vols: pd.DataFrame,
        valuation_date: pd.Timestamp | None = None,
    ) -> None:
        self.discount_curve = discount_curve
        self.credit_spreads = credit_spreads
        self.recovery_rate = recovery_rate
        self.prices = prices
        self.log_returns = log_returns
        self.atm_vols = atm_vols
        self.valuation_date = valuation_date

    @classmethod
    def load(
        cls,
        *,
        discount_curve_path: str | Path,
        credit_data_path: str | Path,
        historical_series_path: str | Path,
        atm_vol_surfaces_path: str | Path,
        valuation_date: pd.Timestamp | str | None = None,
    ) -> "MarketData":
        valuation_date_ts = (
            pd.Timestamp(valuation_date) if valuation_date is not None else None
        )

        discount_curve = cls._load_discount_curve(
            path=discount_curve_path,
            valuation_date=valuation_date_ts,
        )
        credit_spreads, recovery_rate = cls._load_credit_data(credit_data_path)
        prices, log_returns = cls._load_historical_series(historical_series_path)
        atm_vols = cls._load_atm_vol_surfaces(atm_vol_surfaces_path)

        return cls(
            discount_curve=discount_curve,
            credit_spreads=credit_spreads,
            recovery_rate=recovery_rate,
            prices=prices,
            log_returns=log_returns,
            atm_vols=atm_vols,
            valuation_date=valuation_date_ts,
        )

    @staticmethod
    def _load_discount_curve(
        path: str | Path,
        valuation_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        df = pd.read_excel(path, header=1, decimal=",")
        df = df.dropna(axis=1, how="all")

        curve = (
            df[["Maturity Date", "Discount Factor"]]
            .rename(
                columns={
                    "Maturity Date": "date",
                    "Discount Factor": "discount_factor",
                }
            )
            .copy()
        )

        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        curve["discount_factor"] = pd.to_numeric(curve["discount_factor"], errors="coerce")

        curve = (
            curve
            .dropna(subset=["date", "discount_factor"])
            .drop_duplicates(subset="date", keep="first")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if valuation_date is not None:
            curve["t"] = (curve["date"] - valuation_date).dt.days / 365.0

        return curve

    @staticmethod
    def _load_credit_data(path: str | Path) -> tuple[pd.DataFrame, float]:
        df = pd.read_excel(path, sheet_name="CDS", header=None)

        spreads = pd.DataFrame(
            {
                "tenor": df.iloc[2, 2:12].tolist(),
                "spread_bps": df.iloc[3, 2:12].tolist(),
            }
        )

        recovery_rate = float(df.iloc[2, 14])
        return spreads, recovery_rate

    @staticmethod
    def _load_historical_series(
        path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        xls = pd.ExcelFile(path)
        price_frames: list[pd.DataFrame] = []

        for sheet_name in xls.sheet_names:
            raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
            header_row = raw.index[raw.iloc[:, 0].astype(str).eq("Exchange Date")][0]

            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
            df = df[["Exchange Date", "Close"]].copy()
            df = df.rename(columns={"Exchange Date": "date", "Close": sheet_name})

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[sheet_name] = pd.to_numeric(df[sheet_name], errors="coerce")
            price_frames.append(df)

        prices = price_frames[0]
        for df in price_frames[1:]:
            prices = prices.merge(df, on="date", how="outer")

        prices = prices.sort_values("date").reset_index(drop=True)

        log_returns = prices.copy()
        asset_cols = [col for col in prices.columns if col != "date"]
        log_returns[asset_cols] = np.log(prices[asset_cols] / prices[asset_cols].shift(1))

        return prices, log_returns

    @staticmethod
    def _load_atm_vol_surfaces(path: str | Path) -> pd.DataFrame:
        xls = pd.ExcelFile(path)
        frames: list[pd.DataFrame] = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name)

            atm = (
                df[["Date", "ATM"]]
                .rename(columns={"Date": "date", "ATM": "atm_vol"})
                .copy()
            )

            atm["date"] = pd.to_datetime(atm["date"], errors="coerce")
            atm["atm_vol"] = pd.to_numeric(atm["atm_vol"], errors="coerce") / 100.0
            atm["underlying"] = sheet_name

            atm = (
                atm
                .dropna(subset=["date", "atm_vol"])
                .sort_values("date")
                .reset_index(drop=True)
            )

            frames.append(atm)

        return pd.concat(frames, ignore_index=True)

    def get_spot_vector(self, underlyings: list[str]) -> list[float]:
        latest = self.prices.sort_values("date").iloc[-1]
        return [float(latest[u]) for u in underlyings]

    def get_log_return_correlation(self, underlyings: list[str]) -> np.ndarray:
        returns = self.log_returns[underlyings].dropna()
        return returns.corr().to_numpy(dtype=float)

    def get_atm_vol_curves(
        self,
        underlyings: list[str] | None = None,
    ) -> pd.DataFrame:
        data = self.atm_vols.copy()

        if underlyings is not None:
            data = data.loc[data["underlying"].isin(underlyings)].copy()

        data = data.sort_values(["underlying", "date"]).reset_index(drop=True)

        if self.valuation_date is not None:
            data["t"] = (
                (data["date"] - self.valuation_date).dt.days / 365.0
            )

        return data

    def discount_factor(self, t: float) -> float:
        curve = self.discount_curve.dropna(subset=["t", "discount_factor"]).sort_values("t")
        return float(
            np.interp(
                t,
                curve["t"].to_numpy(dtype=float),
                curve["discount_factor"].to_numpy(dtype=float),
            )
        )

    def get_cds_curve(self) -> tuple[list[float], list[float]]:
        tenor_map = {
            "6M": 0.5,
            "1Y": 1.0,
            "2Y": 2.0,
            "3Y": 3.0,
            "4Y": 4.0,
            "5Y": 5.0,
            "7Y": 7.0,
            "10Y": 10.0,
            "15Y": 15.0,
            "20Y": 20.0,
            "30Y": 30.0
        }

        cds = self.credit_spreads.copy()
        cds["tenor_years"] = cds["tenor"].map(tenor_map)
        cds["spread"] = pd.to_numeric(cds["spread_bps"], errors="coerce") / 1e4
        cds = cds.dropna(subset=["tenor_years", "spread"]).sort_values("tenor_years")

        return cds["tenor_years"].tolist(), cds["spread"].tolist()