from datetime import timedelta
import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

### OPTIONS ###
version = "18"
# Use to get a company-year linear trend, set to false for a company trend and a yearly
# trend across all companies.
year_company_data = True
# This is a switch to allow investigation of multiple sets of companies. Valid options
# include:
# "renewable_all", "fossil_34", "control_34" (top 20 of any kind), "constsust_34"
# (top 34 with neutral ethics ratings), "greencred_34", "fossil_100",
# "all_companies" (top 100 with any ethics ratings, uses rating as a continuous variable)
# and a few assorted variables for one-off investigations. Some of these may be desired
# without norm groups.
filetypes = [
    "constsust_20", "constsust_32",
    #"renewable_all",  "fossil_32", "greencred_32",
    #"renewable_20", "fossil_20", "greencred_20",
    #"fossil_100"
]
# If this variable is not empty/False, we switch to studing a different time series.
# Options include None = COPs,
# "OilSpill", "OPEC", "OPEC_conf_28" (only 28 equally spaced OPEC meetings so stats are
# easier), "IPCC"
copOrOther = "OPEC"
# If this is a string (probably "constsust_34"), we subtract the average fractional
# change in this filetype
# before calculating statistics.
norm_group = None #"constsust_32"
# Do we want to only include companies that have data before this date:
require_time_start = pd.to_datetime('2011-01-01')
# Do we want to include a period some days before/after the dictionary of events?
# You will need padding for 1-day events (e.g. OPEC meetings, IPCC reports, OilSpills)
padafter = 0
padbefore = 0
# Do we want to remove data from days with stock splits and low volume?
# (remove days with vol<100 and the day before and after)
skip_dodgy_days = True
# A date before which no data is used (even to establish the normal correlation between
# neutral group and active group
startdate = pd.datetime(year=1995, month=1, day=1)

### Code begins ###

for filetype in filetypes:
    if copOrOther:
        output = f"./output/version{version}/{copOrOther}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
    else:
        output = f"./output/version{version}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
    if skip_dodgy_days:
        output += "_cleaned"
    Path(output).mkdir(exist_ok=True, parents=True)

    # name of the relevant column in the company df. May be overwritten below depending
    # on filetype.
    close = "Close"
    # List of companies whose files we will read
    companylist = os.listdir(f"./input/{filetype}/")
    companylist = [x[:-4] for x in companylist]
    try:
        intendedLen = int(filetype.split("_")[-1])
    except: 
        intendedLen = None
    if intendedLen:
        assert len(companylist) == intendedLen
    
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
        copdates = copdates.iloc[
                   np.round(np.arange(0, len(copdates), (len(copdates) / 28))),
                   :].reset_index()
        meetingstring = f"OPEC meeting"
        copOrOtherLongstring = meetingstring
    elif copOrOther == "OPEC_conf_28":
        copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",",
                               parse_dates=["Date"])
        copdates["Start"] = copdates["Date"]
        copdates["End"] = copdates["Date"]
        copdates = copdates.loc[
            ["Meeting of the OPEC Conference" in i for i in copdates["Meeting Title"]]]
        copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
        copdates = copdates.iloc[
                   np.round(np.arange(0, len(copdates), (len(copdates) / 28))),
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


    # This reads the data from a filestring ending "filetype/company.csv" and appends it
    # to the list "results"
    def read_company_data(filetype, company, results):
        file_path = f'./input/{filetype}/{company}.csv'
        # Read the data from the CSV file into a DataFrame
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
        df = df[np.isfinite(df[close])]
        df["company"] = company
        df["DayVar"] = (df["High"] - df["Low"]) / df["High"]
        df["DayChange"] = df[close].pct_change()
        df["COP"] = np.nan
        for num, row in copdates.iterrows():
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
    all_data = all_data[
        ((all_data["DayChange"] > -0.5) & (all_data["DayChange"] < 0.5))
    ]
    assert all_data.COP.sum() > len(copdates)

    if norm_group:
        normdata = []
        for company in companynormlist:
            read_company_data(norm_group, company, normdata)
        normdata = pd.concat(normdata)
        meanNorm = normdata.groupby('Date').mean()

    if skip_dodgy_days:
        all_data_2 = all_data[
            (all_data["Volume"] > 100) & (all_data["Volume"].shift(1) > 100) &
            (all_data["Volume"].shift(-1) > 100) & (all_data["Stock Splits"] == 0)
            ]
        deleted_data = all_data[
            ~((all_data["Volume"] > 100) & (all_data["Volume"].shift(1) > 100) &
              (all_data["Volume"].shift(-1) > 100) & (all_data["Stock Splits"] == 0))
        ]
        all_data = all_data_2
        if norm_group:
            meanNorm = normdata.groupby('Date').mean()
    all_cat = all_data.copy()
    all_cat = all_cat[all_cat.Date >= startdate]
    if year_company_data:
        all_cat["Year_company"] = all_cat.Date.dt.year.astype(str) + "_" + all_cat[
            "company"
        ]
        del all_cat["company"]
    else:
        all_cat["Year"] = all_cat.Date.dt.year.astype(str)
    all_cat = pd.get_dummies(all_cat, drop_first=True)

    if norm_group:
        all_cat = pd.merge(all_cat,
                           meanNorm.loc[:, ["DayChange", "DayVar"]].reset_index(),
                           on="Date")
    cop_numbers = pd.Series([x if x == x else -1 for x in all_cat["COP"]])
    all_cat["COP"] = [1 if x == x else 0 for x in all_cat["COP"]]
    good_index = (np.isnan(all_cat).sum(axis=1) == 0)
    all_cat = all_cat[good_index]
    all_cat = pd.get_dummies(all_cat)

    # If we have data about ethics ratings we can also add an interaction between this
    # and COPs
    if filetype == "all_companies":
        all_cat["COP_good_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] < 20)
        all_cat["COP_mid_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] > 20) * (
                                                all_cat["Sustainalytics value"] < 30)
        all_cat["COP_bad_ethics"] = all_cat["COP"] * (
                    all_cat["Sustainalytics value"] > 30)
        # Since all companies are covered by one of these, we remove the
        # general case to prevent colinearity issues
        del all_cat["COP"]

    # See if adding the number of the COP implies COPs are getting better or worse the
    # second time
    for yearstring, include_copyear_col in [("", False), ("_COPyear", True)]:
        if not norm_group:
            target = "DayChange"
            ignore_col = "DayVar"
            X_train = all_cat.loc[:, all_cat.columns != ignore_col].iloc[:, 9:]
        else:
            target = "DayChange_x"
            ignore_col = "DayVar_y"
            X_train = all_cat.loc[:, all_cat.columns != ignore_col].iloc[:, 10:]
        if include_copyear_col:
            X_train["COPnumbers"] = cop_numbers[good_index.values]
        # We have established that this works so no longer use the test-train distinction
        y_train = all_cat[target]
        # Fit the results
        ls = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        # Output model results
        summary_str = ls.summary().as_text()
        with open(output + f"/OLSsummary{yearstring}.txt", "w") as f:
            f.write(summary_str)
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
            X2_train = all_cat.loc[
                           accept_index_dayvar, all_cat.columns != ignore_col
                       ].iloc[:, 10:]
        if include_copyear_col:
            X2_train["COPnumbers"] =  cop_numbers[accept_index_dayvar.values].values
        y2_train = all_cat.loc[accept_index_dayvar, target]
        try:
            ls2 = sm.OLS(y2_train, sm.add_constant(X2_train)).fit()
        except Exception as e:
            print(f"failed to resolve for file {filetype} due to exception  {e}")
            ls2 = sm.OLS(y2_train[1:], sm.add_constant(X2_train.iloc[1:, :])).fit()
        summary_str2 = ls2.summary().as_text()
        with open(output + f"/OLSsummaryDailyVar{yearstring}.txt", "w") as f:
            f.write(summary_str2)

    print(f'Finished {filetype}')

