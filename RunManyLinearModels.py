from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp, gstd, kstest, ttest_ind
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

version = "15"
# This is a switch to allow investigation of multiple sets of companies. Valid options include:
# "renewable_all", "fossil_34", "control_34" (top 20 of any kind), "constsust_34"
# (top 34 with neutral ethics ratings), "greencred_34", "fossil_100",
# "all_companies" (top 100 with any ethics ratings, uses rating as a continuous variable)
# and a few assorted variables for one-off investigations.
filetypes = ["renewable_20", "fossil_20", "greencred_20"]
# Do we want to
# Only include companies that have data before this date:
require_time_start = pd.to_datetime('2011-01-01')
# Do we want to include a period some days before/after the dictionary of events?
# You will need padding for 1-day events (e.g. OPEC meetings, IPCC reports, OilSpills)
padafter = 2
padbefore = 0
# If this variable is not empty/False, we switch to studing a different time series. Options include None = COPs,
# "OilSpill", "OPEC", "OPEC_28" (only 28 equally spaced OPEC meetings so stats are easier), "IPCC"
copOrOther = "OPEC_Conference"
# If this is a string (probably "constsust_34"), we subtract the average fractional change in this filetype
# before calculating statistics.
norm_group = "constsust_34"
# Do we want to remove data from days with stock splits and low volume? (remove days with vol<1000 and the day before and after)
skip_dodgy_days = True

for filetype in filetypes:
    if copOrOther:
        output = f"./output/version{version}/{copOrOther}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
    else:
        output = f"./output/version{version}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
    if skip_dodgy_days:
        output += "_cleaned"
    Path(output).mkdir(exist_ok=True, parents=True)

    # name of the relevant column in the company df. May be overwritten below depending on filetype. 
    close = "Close"
    # List of companies whose files we will read
    if filetype == "renewable":
        companylist = [
            "0916.HK", "BEP", "EDPR.LS", "FSLR", "NEE", "NHPC.NS", "SUZLON.NS",
            "VWS.CO", "NPI.TO", "009830.KS"
        ]
    elif filetype == "renewable_20":
        companylist = [
            "0916.HK", "BEP", "EDPR.LS", "FSLR", "NEE", "NHPC.NS", "SUZLON.NS",
            "VWS.CO", "NPI.TO", "009830.KS",
            "ORA", "3800.HK", "PLUG", "NDX1.F", "BLX.TO", "ECV.F", "SLR.MC", "S92.F",
            "VBK.F", "CSIQ",
        ]
    elif filetype == "fossil":
        companylist = [
            "XOM", "CVX", "SHEL", "601857.SS", "TTE", "COP", "BP", "PBR", "EQNR",
            "600028.SS"
        ]
    elif filetype == "fossil_20":
        companylist = [
            "XOM", "CVX", "SHEL", "601857.SS", "TTE", "COP", "BP", "PBR", "EQNR",
            "600028.SS",
            "0883.HK", "SO", "ENB", "SLB", "DUK", "EOG", "CNQ", "EPD", "E", "OXY"

        ]
    elif filetype == "constsust_20":
        companylist = ["ABT", "AMZN", "AZN", "BAC", "BRK-B", "COST", "GOOG", "JNJ",
                       "JPM", "KO",
                       "LLY", "MCD", "MRK", "NESN.SW", "NVO", "PEP", "PG", "ROG.SW",
                       "TYT.L", "WMT"]
    elif filetype == "control":
        companylist = ["AAPL", "AMZN", "BRK-B", "GOOG", "LLY", "MSFT", "NVDA", "TSM",
                       "UNH", "V"]
    elif filetype == "control_20":
        companylist = ["AAPL", "AMZN", "BRK-B", "GOOG", "LLY", "MSFT", "NVDA", "TSM",
                       "UNH", "V",
                       "HD", "PG", "005930.KS", "MC.PA", "JNJ", "MA", "WMT", "AVGO",
                       "NVO", "JPM"]
    elif filetype == "oilprice":
        companylist = ["OilPrice"]
        close = "Adj Close**"
    elif filetype == "tmp":
        companylist = ["^SPX"]
    elif filetype == "oilfuturesApril":
        companylist = ["CrudeOilWTIFrontMonthApril"]
    elif filetype == "greencred_20":
        companylist = ["005930.KS", "AAPL", "ACN", "ADBE", "AMD", "ASML", "AVGO", "CRM",
                       "HD", "MA",
                       "MC.PA", "MSFT", "NFLX", "NVDA", "NVS", "ORCL", "RMS.PA", "TMO",
                       "UNH", "V"]
    elif filetype == "dirty_20":
        companylist = [
            'XOM', '600519.SS', 'CVX', 'KO', 'RELIANCE.NS',
            'SHEL', '601857.SS', 'WFC', '601288.SS', '601988.SS', 'BA', 'COP',
            'RIO', 'PBR', 'BP', '601088.SS', 'EQNR', 'MO', 'CNQ', 'ITC.NS'
        ]
    else:
        companylist = os.listdir(f"./input/{filetype}/")
        companylist = [x[:-4] for x in companylist]
        # In these cases, we expect there to be around 100 items
        if len(companylist) < 21:
            raise ValueError("Invalid filetype option")
    if norm_group:
        companynormlist = os.listdir(f"./input/{norm_group}/")
        companynormlist = [x[:-4] for x in companynormlist]
        try:
            norm_string_end = int(norm_group.split("_")[-1])
        except:
            norm_string_end = None
        if norm_string_end:
            assert len(companynormlist) == norm_string_end
    # Load dates
    if not copOrOther:
        copdates = pd.read_csv("./input/CopDates.txt", delimiter="|")
        copdates = copdates.iloc[:, 1:-1]
        copdates.columns = copdates.columns.str.replace('\s+', '')
        copdates["Start"] = pd.to_datetime(copdates["Start"])
        copdates["End"] = pd.to_datetime(copdates["End"])
        meetingstring = "COP number"
        copOrOtherLongstring = "COP"
    elif copOrOther == "OPEC":
        copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        meetingstring = f"OPEC_meeting"
        copOrOtherLongstring = meetingstring
    elif copOrOther == "OPEC_Conference":
        copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
        copdates = copdates.loc[
            ["Meeting of the OPEC Conference" in i for i in copdates["Meeting Title"]]]
        copdates = copdates.reset_index(drop=True)
        meetingstring = f"OPEC meeting"
        copOrOtherLongstring = meetingstring
    elif copOrOther == "OPEC_28":
        copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
        copdates.iloc[
        np.arange(len(copdates) % 28, len(copdates), (len(copdates) // 28)),
        :].reset_index()
        meetingstring = f"OPEC meeting"
        copOrOtherLongstring = meetingstring
    elif copOrOther == "OilSpill":
        copdates = pd.read_csv("./input/Oil spills data_v2.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        meetingstring = f"Oil spill"
        copOrOtherLongstring = meetingstring
    elif copOrOther == "IPCC":
        copdates = pd.read_csv("./input/IPCC_dates.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        meetingstring = "IPCC report"
        copOrOtherLongstring = "IPCC report release"
    else:
        raise ValueError("Did not specify a valid copOrOther")
    if copOrOther:
        # In all cases, we can't allow duplicate dates so should  flush them out.
        copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
    if padbefore:
        copdates["Start"] = copdates["Start"] - timedelta(days=padbefore)
    if padafter:
        copdates["End"] = copdates["End"] + timedelta(days=padafter)


    # This reads the data from a filestring ending "filetype/company.csv" and appends it to the list results
    def read_company_data(filetype, company, results):
        file_path = f'./input/{filetype}/{company}.csv'  # Name of the variable denoting price at close
        # Read the data from the CSV file into a DataFrame
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
        df = df[np.isfinite(df[close])]
        df["company"] = company
        df["DayVar"] = (df["High"] - df["Low"]) / df["High"]
        df["DayChange"] = df[close].pct_change()
        df["COP"] = np.nan
        for num, row in copdates.iterrows():
            if (
                    row["Start"] > df["Date"].min()
            ) & (row["End"] < df["Date"].max()
            ) & (sum(
                (df["Date"] >= row["Start"]) & (df["Date"] <= row["End"])
            ) > 1):
                df.loc[(row["Start"] <= df["Date"]) & (
                            row["End"] >= df["Date"]), "COP"] = num
        results.append(df)

    all_data = []
    for company in companylist:
        read_company_data(filetype, company, all_data)
    all_data = pd.concat(all_data)

    if any(all_data[all_data.Close < 0]):
        print("Warning: data goes negative")
        print(all_data[all_data.Close < 0])
        all_data = all_data[all_data.Close > 0]
    all_data = all_data[((all_data["DayChange"] > -0.5) & (all_data["DayChange"] < 0.5))]

    if norm_group:
        normdata = []
        for company in companynormlist:
            read_company_data(norm_group, company, normdata)
        normdata = pd.concat(normdata)
        meanNorm = normdata.groupby('Date').mean()

    if skip_dodgy_days:
        all_data_2 = all_data[
            (all_data["Volume"] > 1000) & (all_data["Volume"].shift(1) > 1000) &
            (all_data["Volume"].shift(-1) > 1000) & (all_data["Stock Splits"] == 0)
            ]
        deleted_data = all_data[
            ~((all_data["Volume"] > 1000) & (all_data["Volume"].shift(1) > 1000) &
              (all_data["Volume"].shift(-1) > 1000) & (all_data["Stock Splits"] == 0))
        ]
        all_data = all_data_2
        if norm_group:
            meanNorm = normdata.groupby('Date').mean()
    all_cat = pd.get_dummies(all_data, drop_first=True)
    all_cat["COP"] = [1 if x == x else 0 for x in all_cat["COP"]]

    all_cat = all_cat[all_cat.Date > pd.datetime(year=1995, month=1, day=1)]
    if norm_group:
        all_cat = pd.merge(all_cat,
                           meanNorm.loc[:, ["DayChange", "DayVar"]].reset_index(),
                           on="Date")
    all_cat = all_cat[np.isnan(all_cat).sum(axis=1) == 0]
    all_cat["Year"] = all_cat.Date.dt.year.astype(str)
    all_cat = pd.get_dummies(all_cat)
    # If we have data about ethics ratings we can also add an interaction between this and COPs
    if filetype == "all_companies":
        all_cat["COP_good_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] < 20)
        all_cat["COP_mid_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] > 20) * (
                                                all_cat["Sustainalytics value"] < 30)
        all_cat["COP_bad_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] > 30)
        # Since all companies are covered by one of these, we remove the general case to prevent colinearity issues
        del all_cat["COP"]

    if not norm_group:
        target = "DayChange"
        ignore_col = "DayVar"
        X_train = all_cat.loc[:, all_cat.columns != ignore_col].iloc[:, 9:]
    else:
        target = "DayChange_x"
        ignore_col = "DayVar_y"
        X_train = all_cat.loc[:, all_cat.columns != ignore_col].iloc[:, 10:]
    # We have established that this works so no longer use the test-train distinction
    y_train = all_cat[target]
    # Fit the results
    ls = sm.OLS(y_train, X_train).fit()
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Output model results
    summary_str = ls.summary().as_text()
    with open(output + "/OLSsummary.txt", "w") as f:
        f.write(summary_str)
    # Now do the same for daily variability
    # Now do the same for daily variability
    if not norm_group:
        target = "DayVar"
        ignore_col = "DayChange"
        accept_index_dayvar = (all_cat[target] >= 0) & (all_cat[target] < 0.5)
        X2_train = all_cat.loc[accept_index_dayvar, all_cat.columns != ignore_col].iloc[
                   :, 9:]
    else:
        target = "DayVar_x"
        ignore_col = "DayChange_y"
        accept_index_dayvar = (all_cat[target] >= 0) & (all_cat[target] < 0.5)
        X2_train = all_cat.loc[accept_index_dayvar, all_cat.columns != ignore_col].iloc[
                   :, 10:]
    y2_train = all_cat.loc[accept_index_dayvar, target]
    ls2 = sm.OLS(y2_train, X2_train).fit()
    summary_str2 = ls2.summary().as_text()
    with open(output + "/OLSsummaryDailyVar.txt", "w") as f:
        f.write(summary_str2)
    print(f'Finished {filetype}')

