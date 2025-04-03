from datetime import timedelta, datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import pyfixest.estimation

### OPTIONS ###
version = "23"
# Use to get a company-year linear trend, set to false for a company trend and a yearly
# trend across all companies.
year_company_data = True
# This dictionary maps normalisation groups (neutral market behaviour) to active groups.
# Norm groups:
# If this is a string, we subtract some amount of the average fractional change in this
# filetype (usually constsust_32 or SandP) to make a linear model of the target variable
# before calculating statistics. This controls for normal stock market activity. If
# None,no normal group is used.
# Valid options for filetypes include:
# "renewable_all", "fossil_34", "constsust_34", "greencred_34",
# "renewable_20", "fossil_20", "constsust_20", "greencred_20"
# and a few assorted variables for one-off investigations.
normgroups_and_filetypes_dict = {
    None: [
        "constsust_20",
        "constsust_32",
    ],
   "constsust_32": [
        "renewable_all", "fossil_32", "greencred_32",
        "renewable_20", "fossil_20",
        "greencred_20"
    ],
    "SandP": [
        "SandP_32", "SandPESG_32",
        "renewable_all", "fossil_32", "greencred_32",
        "renewable_20", "fossil_20", "greencred_20"
    ],
}
# If this variable is not empty/False, we switch to studing a different time series.
# Options include None = COPs,
# "OilSpill", "OPEC_Conference", "OPEC_conf_29" (only 29 equally spaced OPEC meetings so stats are
# easier), "IPCC"
copOrOther = "OPEC_conf_29"

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
# If we include inflation as a variable, list it here. We will loop over these.
# Note this loop should have at most one non-None value.
inflation_vars = ["^IRX", None]

### Code begins ###

def rephrase_linear_model_results(ls):
    # Check that the data is good (A few dud rows may be caused by removing scarce data
    assert ls.pvalues.isna().any().sum() < 3
    # Increases the precision of the useful parts of the summary string to print
    ls_summary = ls.summary().as_text().split("\n")[:12] + [
            pd.DataFrame({
                "coef": ls.params,
                "std err": ls.bse,
                "t": ls.tvalues,
                "P>|t|": ls.pvalues,
            }).to_string()
        ]
    return "\n".join(ls_summary)

def usepyfixest(all_data_inc_inflation, norm_group, inflation_var, copyear_on):
    addstring = ""
    if norm_group:
        addstring += " + target_norm "
    if inflation_var:
        addstring += " + inflation "
    if copyear_on :
        addstring += " + COPyear "
    ls_pf = pyfixest.estimation.feols(
        f"target ~ COP{addstring}|company^Year",
        all_data_inc_inflation
    )

    return pd.DataFrame({
        "Coefficient": ls_pf.coef(),
        "Std. Error": ls_pf.se(),
        "t-value": ls_pf.tstat(),
        "P-value": ls_pf.pvalue(),
        "n": ls_pf._N
    })

# Start the actual loops over possible inputs
for norm_group, filetypes in normgroups_and_filetypes_dict.items():
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
            assert len(companynormlist) > 0
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
            copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",", parse_dates=["Start date", "End date"])
            copdates = copdates.rename(columns={"Start date": "Start", "End date": "End"})
            meetingstring = f"OPEC_meeting"
            copOrOtherLongstring = meetingstring
        elif copOrOther == "OPEC_Conference":
            copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",", parse_dates=["Start date", "End date"])
            copdates = copdates.rename(columns={"Start date": "Start", "End date": "End"})
            copdates = copdates.loc[
                ["Meeting of the OPEC Conference" in i for i in copdates["Meeting Title"]]]
            copdates = copdates.reset_index(drop=True)
            meetingstring = f"OPEC meeting"
            copOrOtherLongstring = meetingstring
        elif copOrOther == "OPEC_conf_29":
            copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",", parse_dates=["Start date", "End date"])
            copdates = copdates.rename(columns={"Start date": "Start", "End date": "End"})
            copdates = copdates.loc[["Meeting of the OPEC Conference" in i for i in copdates["Meeting Title"]]]
            # Filter for only weekday events
            copdates = copdates[copdates.Start.dt.weekday < 5]
            # Filter for evenly spaced events
            copdates = copdates.iloc[
                       np.round(np.arange(0, len(copdates)-1, (len(copdates)-1) / 29)), :
                       ].reset_index()
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
            try:
                copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
            except KeyError:
                pass
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

        # optionally load the norm group and filter for reasonable values
        if norm_group:
            normdata = []
            for company in companynormlist:
                read_company_data(norm_group, company, normdata)
            normdata = pd.concat(normdata)
            normdata = normdata[normdata.Close > 0]
            normdata = normdata[
                ((normdata.DayChange > -0.5) & (normdata.DayChange < 0.5))
            ]
            if skip_dodgy_days:
                normdata = normdata[
                    (normdata["Volume"] > 100) & (normdata["Volume"].shift(1) > 100) &
                    (normdata["Volume"].shift(-1) > 100) & (normdata["Stock Splits"] == 0)
                ]
            meanNorm = normdata.groupby('Date').mean()

        if skip_dodgy_days:
            all_data = all_data[
                (all_data["Volume"] > 100) & (all_data["Volume"].shift(1) > 100) &
                (all_data["Volume"].shift(-1) > 100) & (all_data["Stock Splits"] == 0)
                ]

        all_data = all_data[all_data.Date >= startdate]
        all_cat = all_data.copy()
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
                               on="Date", how="left")
            all_cat[["DayChange_y", "DayVar_y"]] = all_cat[["DayChange_y", "DayVar_y"]].fillna(0)
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
        for inflation_var in [v for v in inflation_vars if v]:
            new_infl_var = pd.read_csv(
                "./input/Assorted/" + inflation_var + ".csv", parse_dates=['Date'], dayfirst=True
            )
            new_infl_var = new_infl_var.loc[:, ["Date", "Close"]].rename(columns={"Close": inflation_var})
            # We want values for every date, so interpolate them
            new_infl_var.set_index("Date", inplace=True)
            new_infl_var = new_infl_var.reindex(
                pd.date_range(start=new_infl_var.index.min(),
                              end=new_infl_var.index.max(),
                              freq='D'))
            new_infl_var = new_infl_var.interpolate(method='linear')
            new_infl_var.reset_index(inplace=True)
            new_infl_var.rename(columns={'index': 'Date'}, inplace=True)

            all_cat = pd.merge(all_cat, new_infl_var, on="Date", how="left")

        # See if adding the number of the COP implies COPs are getting better or worse the
        # second time
        for yearstring, include_copyear_col in [("_COPyear", True), ("", False)]:
            for inflation_var in inflation_vars:
                if inflation_var:
                    inflation_exists = np.isfinite(all_cat[inflation_var])
                    yearnormstring = yearstring + "_Norm" + inflation_var
                else:
                    inflation_exists = np.isfinite(all_cat["Date"])
                    yearnormstring = yearstring
                if not norm_group:
                    target = "DayChange"
                    ignore_col = ["DayVar"] + [v for v in inflation_vars if v != inflation_var]
                    X_train = all_cat.loc[inflation_exists, [c for c in all_cat.columns if c not in ignore_col]].iloc[:, 9:]
                else:
                    target = "DayChange_x"
                    ignore_col = ["DayVar_y"] + [v for v in inflation_vars if v != inflation_var]
                    X_train = all_cat.loc[inflation_exists, [c for c in all_cat.columns if c not in ignore_col]].iloc[:, 10:]
                if include_copyear_col:
                    X_train["COPnumbers"] = cop_numbers[good_index.values][inflation_exists]
                # We have established that this works so no longer use the test-train distinction
                y_train = all_cat[target][inflation_exists]
                # Fit the results
                try:
                    ls = sm.OLS(y_train, sm.add_constant(X_train)).fit()
                except Exception as e:
                    print(f"failed to resolve for file {filetype} due to exception  {e}, removing data with fewest results from dummy variable")
                    removal_cols = (X_train!=0).sum() == (X_train!=0).sum().min()
                    assert removal_cols.sum() < 5
                    keep_rows = X_train.loc[:, removal_cols].sum(axis=1)==0
                    print(f"Removing {len(X_train) - keep_rows.sum()} rows")
                    ls = sm.OLS(y_train[keep_rows], sm.add_constant(X_train[keep_rows])).fit()
                # Output model results
                summary_str = rephrase_linear_model_results(ls)
                with open(output + f"/OLSsummary{yearnormstring}.txt", "w") as f:
                    f.write(summary_str)

                # Also perform the fit with pyfixedest to check statistical rigor
                if inflation_var:
                    all_data_inc_inflation = pd.merge(
                        all_data, new_infl_var, on="Date",
                        how="left").rename(columns={inflation_var: "inflation"})
                else:
                    all_data_inc_inflation = all_data.copy()
                if norm_group:
                    all_data_inc_inflation = pd.merge(all_data_inc_inflation,
                                                      meanNorm.loc[:, ["DayChange",
                                                                       "DayVar"]].reset_index(),
                                                      on="Date", how="left").rename(
                        columns={target.replace("x", "y"): "target_norm"})
                all_data_inc_inflation["Year"] = all_data_inc_inflation.Date.dt.year.astype(
                    str)
                all_data_inc_inflation = all_data_inc_inflation.rename(
                    columns={target: "target"})
                if include_copyear_col:
                    all_data_inc_inflation["COPyear"] = -1
                    all_data_inc_inflation.loc[
                        ~all_data_inc_inflation.COP.isna(), "COPyear"
                    ] =all_data_inc_inflation.loc[~all_data_inc_inflation.COP.isna(), "COP"]

                all_data_inc_inflation.loc[
                    ~all_data_inc_inflation.COP.isna(), "COP"] = 1
                all_data_inc_inflation.loc[all_data_inc_inflation.COP.isna(), "COP"] = 0

                # Write the results in a convenient fashion with high precision
                results_df = usepyfixest(all_data_inc_inflation, norm_group, inflation_var, include_copyear_col)

                results_df.to_csv(output + f"/FEOLSdataSummary{target}{yearnormstring}.csv")

                # Now do the same for daily variability
                if not norm_group:
                    target = "DayVar"
                    ignore_col = ["DayChange"] + [v for v in inflation_vars if v != inflation_var]
                    accept_index_dayvar = (all_cat[target] >= 0) & (all_cat[target] < 0.5) & inflation_exists
                    X2_train = all_cat.loc[accept_index_dayvar,
                        [c for c in all_cat.columns if c not in ignore_col]
                    ].iloc[:, 9:]
                else:
                    target = "DayVar_x"
                    ignore_col = ["DayChange_y"] + [v for v in inflation_vars if v != inflation_var]
                    accept_index_dayvar = (all_cat[target] >= 0) & (all_cat[target] < 0.5) & inflation_exists
                    X2_train = all_cat.loc[
                       accept_index_dayvar,
                       [c for c in all_cat.columns if c not in ignore_col]
                    ].iloc[:, 10:]
                if include_copyear_col:
                    X2_train["COPnumbers"] = cop_numbers[accept_index_dayvar.values].values
                y2_train = all_cat.loc[accept_index_dayvar, target]
                try:
                    ls2 = sm.OLS(y2_train, sm.add_constant(X2_train)).fit()
                except Exception as e:
                    print(f"failed to resolve for file {filetype} due to exception for variability {e}")
                    removal_cols = (X2_train != 0).sum() == (X2_train != 0).sum().min()
                    assert removal_cols.sum() < 3
                    keep_rows = X2_train.loc[:, removal_cols].sum(axis=1) == 0
                    print(f"Removing {len(X_train) - keep_rows.sum()} rows")
                    ls2 = sm.OLS(y2_train[keep_rows], sm.add_constant(X2_train.loc[keep_rows, :])).fit()
                summary_str2 = rephrase_linear_model_results(ls2)
                with open(output + f"/OLSsummaryDailyVar{yearnormstring}.txt", "w") as f:
                    f.write(summary_str2)

                # Also perform the fit with pyfixedest to check statistical rigor
                if inflation_var:
                    all_data_inc_inflation = pd.merge(
                        all_data, new_infl_var, on="Date",
                        how="left").rename(columns={inflation_var: "inflation"})
                else:
                    all_data_inc_inflation = all_data.copy()
                if norm_group:
                    all_data_inc_inflation = pd.merge(all_data_inc_inflation,
                        meanNorm.loc[:, ["DayChange", "DayVar"]].reset_index(),
                        on="Date", how="left").rename(columns={target.replace("x", "y"): "target_norm"})
                all_data_inc_inflation["Year"] = all_data_inc_inflation.Date.dt.year.astype(str)
                all_data_inc_inflation = all_data_inc_inflation.rename(columns={target: "target"})
                if include_copyear_col:
                    all_data_inc_inflation["COPyear"] = -1
                    all_data_inc_inflation.loc[
                        ~all_data_inc_inflation.COP.isna(), "COPyear"
                    ] = all_data_inc_inflation.loc[
                        ~all_data_inc_inflation.COP.isna(), "COP"]

                all_data_inc_inflation.loc[
                    ~all_data_inc_inflation.COP.isna(), "COP"] = 1
                all_data_inc_inflation.loc[all_data_inc_inflation.COP.isna(), "COP"] = 0

                # Write the results in a convenient fashion with high precision
                results_df = usepyfixest(all_data_inc_inflation, norm_group, inflation_var, include_copyear_col)
                results_df.to_csv(output + f"/FEOLSdataSummary{target}{yearnormstring}.csv")

        print(f'Finished {filetype}')

