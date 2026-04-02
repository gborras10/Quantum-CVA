from quantum_cva.multi_asset.instruments.market_data import MarketData

market_data = MarketData.load(
    discount_curve_path="data/loaded_market_data/discount_curve.xlsx",
    credit_data_path="data/loaded_market_data/iberdrola_data.xlsx",
    historical_series_path="data/loaded_market_data/time_series.xlsx",
    atm_vol_surfaces_path="data/loaded_market_data/vol_surfaces.xlsx",
    valuation_date="2026-03-15",
)

underlyings = [".STOXX50E", ".FTSE", ".SSMI"]

S0_list = market_data.get_spot_vector(underlyings)
div_yields = [0.0287831, 0.0224722, 0.0316306]
atm_vol_curves = market_data.get_atm_vol_curves(
    underlyings=[".STOXX50E", ".FTSE", ".SSMI"]
)
rho_3d = market_data.get_log_return_correlation(underlyings)

R_cva = 0.415
R_cds = market_data.recovery_rate
lost_given_default = 1.0 - R_cva

cds_tenors_years, cds_spreads = market_data.get_cds_curve()

P0_flat = lambda u: market_data.discount_factor(u)
drift_list = [P0_flat(1) - div for div in div_yields]

print("Spot vector:", S0_list)
print("ATM vol curves:\n", atm_vol_curves)
print("Correlation matrix:\n", rho_3d)
print("CDS tenors (years):", cds_tenors_years)
print("CDS spreads:", cds_spreads)
print("Flat discount factor at 1 year:", P0_flat(1.0))
print("Drift term list:", drift_list)