"""
High-Fidelity Synthetic Dataset Generator for Henry Hub Natural Gas Spot Price Forecasting.
Replicates the statistical properties, correlations, autocorrelation structure,
volatility clustering (heteroskedasticity), and historical extreme events from the paper.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class NaturalGasDatasetGenerator:
    """
    Generates a high-fidelity synthetic dataset matching the specifications of the
    Columbia University LNN Henry Hub natural gas forecasting paper (Jan 6, 2015 to Aug 29, 2025).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.num_samples = 2645
        self.start_date = "2015-01-06"
        self.end_date = "2025-08-29"

    def generate(self) -> pd.DataFrame:
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # 1. Generate Business Days / Trading Days
        # Create a date range of business days, then select exactly the number of samples
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        if len(dates) < self.num_samples:
            # If business days are slightly short, pad them
            dates = pd.date_range(start=self.start_date, periods=self.num_samples, freq="B")
        else:
            dates = dates[:self.num_samples]

        # 2. Simulate Latent Market Regimes and Volatility Clustering
        # We model this via a GARCH-like or Markov-switching process to capture:
        # - heteroskedasticity (volatility clustering)
        # - extreme event spikes (Winter Storm Uri Feb 2021, Ukraine crisis 2022, Winter Storm Elliott Dec 2022, etc.)
        
        # Base volatility of daily returns
        vol = np.zeros(self.num_samples)
        returns = np.zeros(self.num_samples)
        
        # Initial volatility and return
        vol[0] = 2.0
        returns[0] = rng.normal(0, vol[0])

        # GARCH(1,1) style parameters
        omega = 0.15
        alpha = 0.12
        beta = 0.85
        
        for t in range(1, self.num_samples):
            # Base volatility updating
            vol[t] = np.sqrt(omega + alpha * (returns[t-1]**2) + beta * (vol[t-1]**2))
            
            # Draw base return
            returns[t] = rng.normal(0, vol[t])

        # 3. Impose Autocorrelation Structure (ACF Lags 1-8 negative, Lag 19 positive)
        # We apply an ARMA(p, q) filter or direct lag-mixing to force:
        # - Mean reversion in lags 1-8 (negative correlation, peak at lag 3 near -0.112)
        # - EIA Storage Report effect at lag 19 (positive spike near +0.059)
        adjusted_returns = np.zeros(self.num_samples)
        for t in range(20, self.num_samples):
            # AR components for mean-reversion at lag 3 and EIA spike at lag 19
            ar_lag1 = -0.05 * adjusted_returns[t-1]
            ar_lag3 = -0.12 * adjusted_returns[t-3]
            ar_lag5 = -0.04 * adjusted_returns[t-5]
            ar_lag19 = 0.07 * adjusted_returns[t-19]
            
            # Combine GARCH return with AR filter
            adjusted_returns[t] = returns[t] + ar_lag1 + ar_lag3 + ar_lag5 + ar_lag19
        
        # Fill in the first 20 steps
        adjusted_returns[:20] = returns[:20]

        # 4. Inject Historical Extreme Events (Tail-Risk Spikes)
        # Map dates to target times to apply deterministic adjustments for real events:
        # A. Winter Storm Uri: Feb 2021 (around index 1500 to 1550)
        # B. Early 2022 Geopolitical Energy Crisis (around index 1750 to 1820)
        # C. Winter Storm Elliott: Dec 2022 (around index 1980 to 2010)
        # D. April 2024 transient imbalance (around Index 2300)
        # E. November 2024 winter reversal (around Index 2450)
        
        for t in range(self.num_samples):
            current_date = dates[t]
            
            # A. Winter Storm Uri: Feb 12 - Feb 19, 2021
            if current_date.year == 2021 and current_date.month == 2:
                if 10 <= current_date.day <= 18:
                    adjusted_returns[t] += rng.uniform(35.0, 65.0)  # Extreme price spike
                elif 19 <= current_date.day <= 24:
                    adjusted_returns[t] -= rng.uniform(25.0, 45.0)  # Reversal crash
            
            # B. 2022 Russia-Ukraine energy spike (March to June 2022)
            elif current_date.year == 2022 and current_date.month in [3, 4, 5, 6]:
                # Frequent volatile jumps
                if rng.random() < 0.15:
                    adjusted_returns[t] += rng.uniform(15.0, 30.0)
                elif rng.random() < 0.15:
                    adjusted_returns[t] -= rng.uniform(15.0, 25.0)
                    
            # C. Winter Storm Elliott: Dec 21 - Dec 28, 2022
            elif current_date.year == 2022 and current_date.month == 12:
                if 20 <= current_date.day <= 26:
                    adjusted_returns[t] += rng.uniform(20.0, 40.0)
                elif 27 <= current_date.day <= 31:
                    adjusted_returns[t] -= rng.uniform(15.0, 30.0)
            
            # D. April 2024 export imbalance
            elif current_date.year == 2024 and current_date.month == 4 and 10 <= current_date.day <= 18:
                adjusted_returns[t] += rng.uniform(12.0, 25.0)
                
            # E. November 2024 winter risk reversal
            elif current_date.year == 2024 and current_date.month == 11 and 15 <= current_date.day <= 22:
                adjusted_returns[t] += rng.uniform(15.0, 28.0)

        # Scale adjusted returns to match a typical daily return standard deviation of ~5.0%
        raw_std = adjusted_returns.std()
        adjusted_returns = (adjusted_returns / raw_std) * 5.2

        # 5. Reconstruct the Henry Hub Price Level from Returns
        # Return Rt = (Pt - Pt-1)/Pt-1 * 100 => Pt = Pt-1 * (1 + Rt / 100)
        prices = np.zeros(self.num_samples)
        prices[0] = 3.0  # Starting gas price in Jan 2015 ($/MMBtu)
        for t in range(1, self.num_samples):
            # Limit return range to prevent negative prices
            ret = max(adjusted_returns[t], -60.0)
            prices[t] = prices[t-1] * (1.0 + ret / 100.0)
            
            # Guarantee a lower bound on gas price (e.g. $1.20)
            if prices[t] < 1.20:
                prices[t] = 1.20 + rng.uniform(0.01, 0.05)
                # Recalculate return for consistency
                adjusted_returns[t] = ((prices[t] - prices[t-1]) / prices[t-1]) * 100.0

        # 6. Generate Co-moving Exogenous Market Features with Multicollinearity
        # Exogenous indicators should match the correlations in Section III-B:
        # - WTI Crude: Moderate positive correlation with Henry Hub
        # - Treasury yields (2Y, 5Y, 10Y, 20Y): Highly correlated with each other
        # - Dollar Index (USD): Low/negative correlation with commodities, very stable
        # - S&P Energy Index & EQT: Correlate strongly with each other, collapse in 2020, spike in 2022
        # - Nuclear Capacity & Outage: Nuclear outage percentage goes up, residual demand for gas goes up
        
        # Standard Normal Exogenous Latent factors
        macro_factor = np.cumsum(rng.normal(0, 0.02, self.num_samples))
        energy_factor = np.cumsum(rng.normal(0, 0.03, self.num_samples))
        
        # WTI Crude Spot Price
        wti = 50.0 + 15.0 * energy_factor + 8.0 * macro_factor + rng.normal(0, 1.5, self.num_samples)
        wti = np.clip(wti, 20.0, 120.0)
        
        # Treasury yield curve (correlated maturities)
        base_rate = 2.0 + 1.5 * macro_factor
        treasury_2y = base_rate + rng.normal(0, 0.15, self.num_samples)
        treasury_5y = base_rate + 0.3 + rng.normal(0, 0.10, self.num_samples)
        treasury_10y = base_rate + 0.6 + rng.normal(0, 0.08, self.num_samples)
        treasury_20y = base_rate + 0.9 + rng.normal(0, 0.12, self.num_samples)
        
        # Clip treasury rates above 0%
        treasury_2y = np.clip(treasury_2y, 0.05, 6.0)
        treasury_5y = np.clip(treasury_5y, 0.10, 6.2)
        treasury_10y = np.clip(treasury_10y, 0.15, 6.5)
        treasury_20y = np.clip(treasury_20y, 0.20, 6.8)
        
        # Treasury Spreads (Exogenous features)
        spread_10y_2y = treasury_10y - treasury_2y
        
        # U.S. Dollar Index (Stable, negatively correlated with commodities)
        usd_index = 95.0 - 5.0 * macro_factor + rng.normal(0, 0.5, self.num_samples)
        usd_index = np.clip(usd_index, 80.0, 115.0)
        
        # S&P Energy Index (Equity Index)
        sp_energy = 450.0 + 120.0 * energy_factor - 30.0 * macro_factor + rng.normal(0, 8.0, self.num_samples)
        sp_energy = np.clip(sp_energy, 200.0, 850.0)
        
        # Dow Jones U.S. Coal Index (Lost value early, recovered slightly in 2022)
        coal_index = 100.0 - 45.0 * (dates.year.values - 2015) / 10.0 + 20.0 * energy_factor + rng.normal(0, 4.0, self.num_samples)
        coal_index = np.clip(coal_index, 10.0, 250.0)
        
        # EQT Stock Price (Tracks S&P Energy and Gas prices closely)
        eqt_price = 25.0 + 0.04 * sp_energy + 3.0 * prices + rng.normal(0, 1.2, self.num_samples)
        eqt_price = np.clip(eqt_price, 5.0, 65.0)
        
        # U.S. Nuclear Outage variables
        # Seasonality-driven, capacity is stable (around 95,000 MW)
        installed_nuc_mw = np.ones(self.num_samples) * 95000.0 + rng.normal(0, 200.0, self.num_samples)
        
        # Outage capacity has strong seasonal peaks (spring and autumn maintenance)
        day_of_year = dates.dayofyear.values
        seasonal_outage_prob = 0.05 + 0.25 * (np.sin(2.0 * np.pi * day_of_year / 365.25 - np.pi/2) ** 4)
        outage_nuc_mw = installed_nuc_mw * seasonal_outage_prob + rng.normal(0, 800.0, self.num_samples)
        outage_nuc_mw = np.clip(outage_nuc_mw, 1000.0, 35000.0)
        
        outage_nuc_pct = (outage_nuc_mw / installed_nuc_mw) * 100.0
        
        # Create exogenous DataFrame
        df = pd.DataFrame(index=dates)
        df["Date"] = dates
        df["Spot Price"] = prices
        df["Spot Return"] = adjusted_returns
        df["Spot Return (AR1)"] = df["Spot Return"].shift(1).fillna(0.0)
        
        # Exogenous indicators (Totaling 30 predictor features with transformations)
        df["WTI Price"] = wti
        df["Treasury_2Y"] = treasury_2y
        df["Treasury_5Y"] = treasury_5y
        df["Treasury_10Y"] = treasury_10y
        df["Treasury_20Y"] = treasury_20y
        df["Treasury_Spread_10Y_2Y"] = spread_10y_2y
        df["USD_Index"] = usd_index
        df["SP_Energy"] = sp_energy
        df["Coal_Index"] = coal_index
        df["EQT_Price"] = eqt_price
        df["Nuclear_Capacity"] = installed_nuc_mw
        df["Nuclear_Outage"] = outage_nuc_mw
        df["Nuclear_Outage_Pct"] = outage_nuc_pct
        
        # Rolling statistical features to reach exactly 30 predictors as used in standard neural models
        df["Spot_Return_Vol_10"] = df["Spot Return"].rolling(10).std().bfill()
        df["Spot_Return_Vol_30"] = df["Spot Return"].rolling(30).std().bfill()
        df["WTI_Return"] = df["WTI Price"].pct_change().fillna(0.0) * 100.0
        df["WTI_Vol_10"] = df["WTI_Return"].rolling(10).std().bfill()
        df["USD_Return"] = df["USD_Index"].pct_change().fillna(0.0) * 100.0
        df["USD_Vol_10"] = df["USD_Return"].rolling(10).std().bfill()
        df["SP_Energy_Return"] = df["SP_Energy"].pct_change().fillna(0.0) * 100.0
        df["EQT_Return"] = df["EQT_Price"].pct_change().fillna(0.0) * 100.0
        
        # Calendar time features
        df["Month_Sin"] = np.sin(2.0 * np.pi * dates.month.values / 12.0)
        df["Month_Cos"] = np.cos(2.0 * np.pi * dates.month.values / 12.0)
        df["Day_Sin"] = np.sin(2.0 * np.pi * dates.dayofweek.values / 5.0)
        df["Day_Cos"] = np.cos(2.0 * np.pi * dates.dayofweek.values / 5.0)
        
        # Spread variables and additional lags to make exactly 30 predictors
        df["WTI_Gas_Spread"] = df["WTI Price"] - df["Spot Price"]
        df["Coal_Gas_Ratio"] = df["Coal_Index"] / (df["Spot Price"] + 1e-4)
        df["Nuclear_Outage_Cap_Ratio"] = df["Nuclear_Outage"] / (df["Nuclear_Capacity"] + 1e-4)
        df["USD_WTI_Ratio"] = df["USD_Index"] / (df["WTI Price"] + 1e-4)
        df["EQT_SP_Ratio"] = df["EQT_Price"] / (df["SP_Energy"] + 1e-4)
        
        # 30 standard neural features
        self.predictor_cols = [
            "Spot Return (AR1)", "WTI Price", "Treasury_2Y", "Treasury_5Y", "Treasury_10Y", "Treasury_20Y",
            "Treasury_Spread_10Y_2Y", "USD_Index", "SP_Energy", "Coal_Index", "EQT_Price",
            "Nuclear_Capacity", "Nuclear_Outage", "Nuclear_Outage_Pct", "Spot_Return_Vol_10",
            "Spot_Return_Vol_30", "WTI_Return", "WTI_Vol_10", "USD_Return", "USD_Vol_10",
            "SP_Energy_Return", "EQT_Return", "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos",
            "WTI_Gas_Spread", "Coal_Gas_Ratio", "Nuclear_Outage_Cap_Ratio", "USD_WTI_Ratio"
        ]
        
        # The standard 30 predictor columns
        assert len(self.predictor_cols) == 30, f"Expected 30 features, got {len(self.predictor_cols)}"
        
        return df


def test_generator():
    gen = NaturalGasDatasetGenerator()
    df = gen.generate()
    print("Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target variable description:\n{df['Spot Return'].describe()}")
    
    # Calculate ACF to verify autocorrelation structure
    acf = [df['Spot Return'].autocorr(lag=i) for i in range(1, 21)]
    print("\nAutocorrelation Function of Returns:")
    for lag, val in enumerate(acf, 1):
        print(f"  Lag {lag}: {val:.4f}")
    
    # Check for Winter Storm Uri spike
    uri_slice = df[(df.index.year == 2021) & (df.index.month == 2)]
    print(f"\nWinter Storm Uri slice (Feb 2021):\n{uri_slice[['Spot Price', 'Spot Return']].head(10)}")


if __name__ == "__main__":
    test_generator()
