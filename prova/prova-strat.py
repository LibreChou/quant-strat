#
#  PROVA-1 Strategy
#  2-factor: Profitability & Valuation 
#  @author:  Jonathan Liang
#
#  Factors:  EV / EBITDA & ROIC, where:
#       EV = [Market value + ((debt in current liabilities) + (long term debt total) + (Preferred/Preference Stock (Capital))] - (Cash and Short-Term Investments)
#       ROIC = (Net Income - Dividends)  / (Total Debt + Total Equity)
#
#  Purpose: balance highly profitable companies with lower valuations
#     (i.e. good profits at a lower price)
#

import scipy as sp
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
#import matplotlib.pyplot as plt


basedir   = "/Users/liangjh/Workspace/quant-asam/prova/"
inputdir  = basedir + "input/"
outputdir = basedir + "output/" 
allocation_buckets = 5

#  READ FUNDAMENTALS DATA
print("Reading Fundamentals Data...")
fundamentals_dt = pd.read_csv(inputdir + "fundamentals-annual.csv", header = 0)

#
#  DATA CLEANUP
#

fundamentals_dt.columns
fundamentals_dt = fundamentals_dt[fundamentals_dt["fic"] == "USA"]  # remove foreign entities
# Make calculations easier, avoid divbyzero
fundamentals_dt = fundamentals_dt[np.logical_and(fundamentals_dt["ebitda"] > 0, 
                                                 fundamentals_dt["seq"] != 0)]

#
#  CALCULATE FACTORS
#
print("Calculating Factors...")
#  Calculate market value (given MV may be missing a lot of data)
fundamentals_dt["market_value"] = fundamentals_dt.apply(lambda row: row["csho"] * row["prcc_f"], axis = 1)
#  Calculate EV and EV/EBITDA (valuation factor)
fundamentals_dt["ev"] = fundamentals_dt.apply(lambda row: (row["market_value"] + row["dlc"] + row["dltt"] + row["pstk"] - row["che"]), axis = 1)
fundamentals_dt["ev_ebitda"] = fundamentals_dt.apply(lambda row: (row["ev"] / row["ebitda"]), axis = 1)
#  Calculate ROIC (profitability factor)
fundamentals_dt["roic"] = fundamentals_dt.apply(lambda row: ((row["ni"] - row["dvt"]) / 
                                                             (row["seq"] + row["dlc"] + row["dltt"])), axis = 1)

#
#  RANKINGS & PORTFOLIO ALLOCATIONS
#  For each fiscal year
#  1. Take highest quintile for ROIC (high profitability)
#  2. Take lowest EV/EBITDA (within ranking [1])
#  3. Output investment decisions to CSV
#

print("Determine Portfolio / Investment Allocations...")
portfolio_allocations_dt = pd.DataFrame()
for year in fundamentals_dt["fyear"].unique():

    curr_dt = fundamentals_dt.loc[fundamentals_dt["fyear"] == year]
    before_cardinality = curr_dt.shape[0]

    #  ROIC (profitability) ranking: select quantile 5 (top quantile)
    quantile_roic = pd.qcut(curr_dt["roic"], allocation_buckets, labels = list(range(1, allocation_buckets + 1)))
    curr_dt["quantile_roic"] = quantile_roic
    curr_dt = curr_dt.loc[curr_dt["quantile_roic"] == allocation_buckets]

    #  EV/EBITDA (valuation) ranking: select quantile 1 (lowest quantile), within ROIC quantile 5
    quantile_ev_ebitda = pd.qcut(curr_dt["ev_ebitda"], allocation_buckets, labels = list(range(1, allocation_buckets + 1)))
    curr_dt["quantile_ev_ebitda"] = quantile_ev_ebitda
    curr_dt = curr_dt.loc[curr_dt["quantile_ev_ebitda"] == 1]

    print("  // Quantiling for year: [" + str(year) + "].  Cardinality: [before: " + str(before_cardinality) + ", after: " + str(curr_dt.shape[0]) + "]")
    portfolio_allocations_dt = portfolio_allocations_dt.append(curr_dt)

#  The investment year (iyear) is one year following the reported fiscal year
portfolio_allocations_dt["iyear"] = portfolio_allocations_dt["fyear"] + 1
portfolio_allocations_dt.rename(columns = {"LPERMNO": "PERMNO", "tic": "ticker"}, inplace = True)

#  Clean up raw data
del fundamentals_dt

#  Write values to output
portfolio_allocations_dt = portfolio_allocations_dt.sort_values(by = ["fyear"])
portfolio_allocations_dt = portfolio_allocations_dt.loc[:, ["fyear", "iyear", "ticker", "PERMNO", 
                                                            "cusip", "gsector", "gind", "market_value", "roic", "ev_ebitda", 
                                                            "quantile_roic", "quantile_ev_ebitda"]]
portfolio_allocations_dt.to_csv(outputdir + "data-portfolio_allocations.csv")


#
#  RETURNS 
#  Derive monthly returns, given annual portfolio allocations
#  We are assuming an equally-weighted portfolio, so we can take the average 
#   return for all allocations for this given period
#

print("Reading / Processing Returns...")
returns_dt = pd.read_csv(inputdir + "returns.csv", header = 0)
returns_dt.rename(columns = {"TICKER": "ticker"}, inplace = True)
returns_dt["month"] = returns_dt.apply(lambda row: int(row["date"] / 100), axis = 1)
returns_dt["iyear"] = returns_dt.apply(lambda row: int(row["date"] / 10000), axis = 1)

#  Merge / combine portfolio allocations with returns
returns_set_dt = portfolio_allocations_dt.merge(returns_dt, on = ["iyear", "PERMNO"], how = "inner")
returns_set_dt = returns_set_dt.convert_objects(convert_numeric = True)

# Clean up giant dataset
del returns_dt

#
#  Derive month-by-month returns, aggregating all individual member stock returns for each month
#       (since we are equally allocating, we just need average returns)
#

# Monthly returns
print("Aggregating Monthly Returns...")
ret_month_group = returns_set_dt.groupby("month")
returns_monthly_dt = ret_month_group["RET"].agg([np.mean, np.std, len])
returns_monthly_dt["month"] = returns_monthly_dt.index
returns_monthly_dt.rename(columns = {"mean": "portfolio_return"}, inplace = True)


#
#  BENCHMARKS
#  via:  S&P 500, CAPM, and FF-3 
# 

print("Loading Benchmarks (SP500, FF-3)...")
sp500_returns_monthly_dt = pd.read_csv(inputdir + "sp500_returns.csv", header = 0)
returns_monthly_bm_dt = returns_monthly_dt.merge(sp500_returns_monthly_dt, on = ["month"], how = "inner")
returns_monthly_bm_dt.rename(columns = {"close": "sp500_close", "return": "sp500_return"}, inplace = True)
returns_monthly_bm_dt["excess_sp500_return"] = (returns_monthly_bm_dt["portfolio_return"] - returns_monthly_bm_dt["sp500_return"])

#  Fama-French 3-Factor Model Data (from Ken French data library)
ff3_dt = pd.read_csv(inputdir + "ff3_research_data_factors.csv", header = 0)
ff3_dt.rename(columns = {"Mkt-RF": "Mkt_RF"}, inplace = True)
ff3_dt = ff3_dt.convert_objects(convert_numeric = True)
ff3_dt["Mkt_RF"] = ff3_dt["Mkt_RF"] / 100.0
ff3_dt["SMB"] = ff3_dt["SMB"] / 100.0
ff3_dt["HML"] = ff3_dt["HML"] / 100.0
ff3_dt["RF"]  = ff3_dt["RF"] / 100.0

returns_monthly_bm_dt = returns_monthly_bm_dt.merge(ff3_dt, on = ["month"], how = "inner")
returns_monthly_bm_dt["excess_rf_return"] = (returns_monthly_bm_dt["portfolio_return"] - returns_monthly_bm_dt["RF"])
returns_monthly_bm_dt.to_csv(outputdir + "data-returns_monthly.csv")

#
#  Annual Returns
#  (+ also relatively to benchmark)
#

#  Prep data for calculations (1 + ret.monthly)
print("Aggregating Annual Returns...")
returns_monthly_bm_dt["portfolio_return_p1"] = (returns_monthly_bm_dt["portfolio_return"] + 1.0)
returns_monthly_bm_dt["excess_sp500_return_p1"]  = (returns_monthly_bm_dt["excess_sp500_return"] + 1.0)
returns_monthly_bm_dt["sp500_return_p1"] = (returns_monthly_bm_dt["sp500_return"] + 1.0)
returns_monthly_bm_dt["iyear"] = returns_monthly_bm_dt.apply(lambda row: int(row["month"] / 100), axis = 1)

# Aggregate calcs
group_annual = returns_monthly_bm_dt.groupby(["iyear"])
returns_annual_bm_dt = group_annual.agg({"portfolio_return_p1": "prod", "excess_sp500_return_p1": "prod", "sp500_return_p1": "prod"})
returns_annual_bm_dt["portfolio_return"] = (returns_annual_bm_dt["portfolio_return_p1"] - 1.0)
returns_annual_bm_dt["excess_sp500_return"] = (returns_annual_bm_dt["excess_sp500_return_p1"] - 1.0)
returns_annual_bm_dt["sp500_return"] = (returns_annual_bm_dt["sp500_return_p1"] - 1.0)
returns_annual_bm_dt.to_csv(outputdir + "data-returns_annual.csv")

#
#  RISK ANALYTICS
#  Quantify beta / alpha under CAPM and FF-3 Models 
#

#  CAPM and FF-3 Model fits
print("Fitting CAPM, FF-3 Models...")
capm_fit = sm.ols(formula = "excess_rf_return ~ Mkt_RF", data = returns_monthly_bm_dt).fit()
ff3_fit  = sm.ols(formula = "excess_rf_return ~ Mkt_RF + SMB + HML", data = returns_monthly_bm_dt).fit()
capm_fit.summary()
ff3_fit.summary()

with open(outputdir + "data-capm_fit.csv", "a") as file:
    file.write(capm_fit.summary().as_csv())
with open(outputdir + "data-ff3_fit.csv", "a") as file:
    file.write(ff3_fit.summary().as_csv())


#
# Industry & Sector Concentrations
# For each investment year, quantify industry & sector concentrations
# Does the strategy naturally skew towards specific industry types?
#

print("Industry & Sector Concentrations...")
group_sector = portfolio_allocations_dt.groupby(["iyear","gsector"])
sector_concentrations_dt = group_sector["market_value"].agg([np.mean, np.sum, len])
sector_concentrations_dt.rename(columns = {"mean": "avg_market_value", "sum": "total_market_value", "len": "count"}, inplace = True)
sector_concentrations_dt.to_csv(outputdir + "data-sector_concentrations.csv")

group_industry = portfolio_allocations_dt.groupby(["iyear", "gind"])
industry_concentrations_dt = group_industry["market_value"].agg([np.mean, np.sum, len])
industry_concentrations_dt.rename(columns = {"mean": "avg_market_value", "sum": "total_market_value", "len": "count"}, inplace = True)
industry_concentrations_dt.to_csv(outputdir + "data-industry_concentrations.csv")

#
# Market Value Distributions
# Using NYSE market cap deciles, we can see how the share targets are distributed
#

print("Market Value / Marketcap Concentrations (by year)..")
group_iyear = portfolio_allocations_dt.groupby(["iyear"])
iyear_stats_dt = group_iyear["market_value"].agg([np.mean, np.std, np.sum, len])
iyear_stats_dt.rename(columns = {"mean": "marketcap_mean", "std": "marketcap_std", "sum": "marketcap_total"}, inplace = True)
iyear_stats_dt.to_csv(outputdir + "data-iyear_marketcap_concentrations.csv")



