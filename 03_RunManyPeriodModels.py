from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.stats import gstd, kstest, ttest_ind, kstwo
import statsmodels.api as sm

### OPTIONS ###

version = "26"
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
    "constsust_32": [
        "renewable_all", "fossil_32", "greencred_32",
        "renewable_20", "fossil_20",
        "greencred_20"
    ],
    None: [
        "constsust_32",
        "constsust_20",
    ],
    "SandP": [
        "SandPESG_32", "SandP_32",
        "renewable_all", "fossil_32", "greencred_32",
        "renewable_20", "fossil_20", "greencred_20"
    ]
}
# Do we want to only subtract a fractional amount of the norm group or the whole thing?
# Use True for only subtracting the correlated fraction, False for subtracting the
# whole of it.
norm_group_fraction = False
# Do we want to plot each company's individual behaviour?
plots_on = True
require_time_start = pd.to_datetime('2011-01-01')
# Do we want to include a period some days before/after the dictionary of events?
# You will need padding for 1-day events (e.g. OPEC meetings, IPCC reports, OilSpills)
padafter = 0
padbefore = 0
# If this variable is not empty/False, we switch to studing a different time series.
# Options include None = COPs, "OilSpill", "OPEC",
# "OPEC_conf_29" (only 29 equally spaced OPEC meetings so stats are easier), "OPEC_Conference"
# (only main conferences, not side meetings), "IPCC"
copOrOther = None
# Number of weeks before and after COP to capture the whole year of counterfactuals.
# Note this must be even.
maxweek = 24
# name of the relevant column in the company df. May be overwritten below depending on filetype.
close = "Close"

### BEGIN CODE ###

# This code reads the company data and inserts it into the lists aggregate_data and all_data.
# It does not return anything.
# filetype is the folder with company data,
# company is the name of the company data file,
# maxweek is how many weeks before and after the COPs (or similar events) to consider the events.
def read_company_data(filetype, company, maxweek, aggregate_data, all_data, copdates):
    file_path = f'./input/{filetype}/{company}.csv'  # Name of the variable denoting price at close

    # Read the data from the CSV file into a DataFrame
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[df["Open"] != "-"]
    df.iloc[:, 1:-1] = df.iloc[:, 1:-1].astype(float)
    if len(df[df[close] <= 0]) > 0:
        print("Contains negative values!")
        df = df[df[close] > 0]
    assert df[
               "Date"].min() < require_time_start, f"Company {company} starts at {df['Date'].min()}, after {require_time_start}"
    df = df[np.isfinite(df[close])]
    # Label periods that are within COP. We consider the day after the close of COP to be included, since it is likely to end
    # on a weekend, and markets will close before final announcements are available.
    # We therefore go from the close of the market day nearest before to the close of the market day after COP.
    offset_results = {}
    for offset in [7 * x for x in range(-maxweek, maxweek + 1, 2)]:
        copChange = []
        for num, row in copdates.iterrows():
            if (
                    row["Start"] + pd.Timedelta(days=offset) > df["Date"].min()
            ) & (row["End"] + pd.Timedelta(days=offset) < df["Date"].max()
            ) & (sum(
                (df["Date"] >= row["Start"] + pd.Timedelta(days=offset)) & (
                        df["Date"] <= row["End"] + pd.Timedelta(days=offset))
            ) > 1):
                copChange.append({
                    "COP": num,
                    "Before": df.loc[
                        df["Date"] == max(df.loc[
                                              df["Date"] < row["Start"] + pd.Timedelta(
                                                  days=offset), "Date"]), close
                    ].iat[0],
                    "After": df.loc[
                        df["Date"] == min(df.loc[df["Date"] > row["End"] + pd.Timedelta(
                            days=offset), "Date"]), close
                    ].iat[0],
                    "SD": df.loc[
                        (df["Date"] >= row["Start"] + pd.Timedelta(days=offset))
                        & ((df["Date"] <= row["End"] + pd.Timedelta(days=offset))),
                        close
                    ].std(),
                    "geoSD": gstd(df.loc[
                                      (df["Date"] >= row["Start"] + pd.Timedelta(
                                          days=offset))
                                      & (df["Date"] <= row["End"] + pd.Timedelta(
                                          days=offset)),
                                      close
                                  ]),
                })
        copChange = pd.DataFrame(copChange)
        copChange["diff"] = (copChange["After"] - copChange["Before"]) / copChange[
            "Before"]
        offset_results[offset / 7] = copChange

    # Analyse the differences
    x = []
    ymean = []
    ystd = []
    yquant = {}
    ystdmean = []
    ygeostdmean = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for key, val in offset_results.items():
        x.append(key)
        ymean.append(val["diff"].mean())
        ystd.append(val["diff"].std())

        ystdmean.append(val["SD"].mean())
        ygeostdmean.append(val["geoSD"].mean())
        yquant[key] = np.quantile(val["diff"], quantiles)

    ymean = np.array(ymean)
    ystd = np.array(ystd)
    yquant = pd.DataFrame(yquant, index=quantiles)
    ystdmean = np.array(ystdmean)
    aggregate_data.append(
        pd.DataFrame({
            "company": company, "x": x, "ymean": ymean,
            "ystd": ystd, "ystdmean": ystdmean, "ygeostdmean": ygeostdmean,
            "y0.1": yquant.loc[0.1, :], "y0.25": yquant.loc[0.25, :],
            "y0.5": yquant.loc[0.5, :], "y0.75": yquant.loc[0.75, :],
            "y0.9": yquant.loc[0.9, :]
        })
    )
    for key, val in offset_results.items():
        val["offset"] = key
        val["company"] = company
        all_data.append(val)
    # No return, modified input instead

# Start the actual loops over possible inputs
for norm_group, filetypes in normgroups_and_filetypes_dict.items():
    if norm_group:
        companynormlist = os.listdir(f"./input/{norm_group}/")
        companynormlist = [x[:-4] for x in companynormlist]
        def rowsubtract(df1, df2, frac, cols):
            # A function to subtract the baseline data from each company for each date.
            # df1 is the baseline data,
            # df2 is the normalisation data (cannot have a "company" column)
            # cols is a list of columns that should be joined on
            df = pd.merge(
                df1.reset_index(), df2, on=cols, suffixes=("_raw", "_norm")
            ).set_index("index")
            for col in [c for c in df1.columns if (c not in cols) and c != "company"]:
                df[col] = df[col + "_raw"] - frac * df[col + "_norm"]
                df = df.drop([col + "_raw", col + "_norm"], axis=1)
            return df
        all_norm_data = []
        aggregate_data_norm = []
    for filetype in filetypes:
        print(f"Starting analysis for {filetype}")
        if copOrOther:
            output = f"./output/version{version}/{copOrOther}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
        else:
            output = f"./output/version{version}/{filetype}/before{padbefore}_after{padafter}_norm{norm_group}"
        if not norm_group_fraction:
            output += "_1"
        Path(output).mkdir(exist_ok=True, parents=True)

        # List of companies whose files we will read
        companylist = os.listdir(f"./input/{filetype}/")
        companylist = [x[:-4] for x in companylist]
        if filetype[-2:] == "20":
            assert len(companylist) == 20
        elif filetype[-2:] == "32":
            assert len(companylist) == 32

        if not copOrOther:
            copdates = pd.read_csv("./input/CopDates.txt", delimiter="|")
            copdates = copdates.iloc[:, 1:-1]
            copdates.columns = copdates.columns.str.replace('\s+', '')
            copdates["Start"] = pd.to_datetime(copdates["Start"])
            copdates["End"] = pd.to_datetime(copdates["End"])
            meetingstring = "COP number"
            copOrOtherLongstring  = "COP"
        elif copOrOther == "OPEC":
            copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",", parse_dates=["Start date", "End date"])
            copdates = copdates.rename(columns={"Start date": "Start", "End date": "End"})
            meetingstring = f"OPEC meeting"
            copOrOtherLongstring = meetingstring
        elif copOrOther == "OPEC_Conference":
            copdates = pd.read_csv("./input/OPEC_all_2002.csv", delimiter=",", parse_dates=["Start date", "End date"])
            copdates = copdates.rename(columns={"Start date": "Start", "End date": "End"})
            copdates = copdates.loc[["Meeting of the OPEC Conference" in i for i in copdates["Meeting Title"]]]
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
            copdates = pd.read_csv("./input/Oil spills data_v2.csv", delimiter=",", parse_dates=["Date"])
            copdates["Start"] = copdates["Date"]
            copdates["End"] = copdates["Date"]
            meetingstring = f"Oil spill"
            copOrOtherLongstring = meetingstring
        elif copOrOther == "IPCC":
            copdates = pd.read_csv("./input/IPCC_dates.csv", delimiter=",", parse_dates=["Date"])
            copdates["Start"] = copdates["Date"]
            copdates["End"] = copdates["Date"]
            meetingstring = "IPCC report"
            copOrOtherLongstring = "IPCC report release"
        else:
            raise ValueError("Did not specify a valid copOrOther")
        if copOrOther:
            # In all cases, we can't allow duplicate dates so should flush them out.
            try:
                copdates = copdates.drop_duplicates(subset=["Date"], keep="first")
            except KeyError:
                pass

        if padbefore:
            copdates["Start"] = copdates["Start"] - timedelta(days=padbefore)
        if padafter:
            copdates["End"] = copdates["End"] + timedelta(days=padafter)

        # Loop over companies recording behaviour during COP or the same time gap shifted
        # by some weeks
        if norm_group:
            if len(all_norm_data) == 0:
                for company in companynormlist:
                     read_company_data(norm_group, company, maxweek, aggregate_data_norm, all_norm_data, copdates)
                all_norm_data = pd.concat(all_norm_data)
                allnormsummarydata = pd.concat(aggregate_data_norm).groupby("x").mean()
                all_norm_data_grouped = all_norm_data.groupby(["COP", "offset"]).mean()

        # Read data for each company
        aggregate_data = []
        all_data = []
        for company in companylist:
            read_company_data(filetype, company, maxweek, aggregate_data, all_data, copdates)
        all_data = pd.concat(all_data)
        alldata = pd.concat(aggregate_data)
        if norm_group:
            if norm_group_fraction:
                crosstable = pd.merge(all_data, all_norm_data, on=["COP", "offset"])
                model = sm.OLS(crosstable["diff_x"], sm.add_constant(crosstable["diff_y"]))
                results = model.fit()
                normfrac = results.params["diff_y"]
            else:
                normfrac = 1
            alldata = rowsubtract(alldata, allnormsummarydata, normfrac, ["x"])
            all_data = rowsubtract(all_data, all_norm_data_grouped, normfrac,
                                   ["COP", "offset"])
        compav = alldata.groupby("x").mean()
        compmed = alldata.groupby("x").median()

        # Continue generating statistics
        definitelynotcops = all_data.loc[
                (all_data["offset"]!=0)&(all_data["offset"]!=1)&(all_data["offset"]!=-1)&(
                    all_data["offset"]!=2)&(all_data["offset"]!=-2)&(all_data["offset"]!=3)&(all_data["offset"]!=-3),
            :]
        distofcops = np.quantile(
            definitelynotcops["diff"],
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )
        distofcopsGeoSD = np.quantile(
            definitelynotcops["geoSD"],
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )
        distofcops = [i*100 for i in distofcops]
        copnum = len(copdates)

        # Plot the results
        if plots_on:
            weeksafterstr = f"Weeks after {copOrOther + ' meeting' if copOrOther else 'COP'} starts"
            plt.plot(compav.index, compmed.ymean*100)
            plt.fill_between(
                compav.index, compav.loc[:, "y0.25"]*100, compav.loc[:, "y0.75"]*100, alpha=0.3, color="red"
            )
            plt.fill_between(
                compav.index, compav.loc[:, "y0.9"]*100, compav.loc[:, "y0.1"]*100, alpha=0.2, color="orange"
            )
            plt.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
            plt.ylabel("Average change during period (%)")
            plt.xlabel(weeksafterstr)
            plt.legend(["Trend", "25-75% range", "10-90% range"])
            plt.savefig(f"{output}/AverageChange.png")

            plt.plot(compmed.index, compmed.ygeostdmean, linewidth=2)
            plt.ylabel("Average geometric SD of companies")
            plt.xlabel(weeksafterstr)
            for c in alldata.company.unique():
                plt.plot(
                    alldata.loc[alldata.company==c, "x"],
                    alldata.loc[alldata.company==c, "ygeostdmean"],
                    alpha=0.2, linewidth=1
                )
            plt.axvline(x=0, color='grey', linewidth=1, alpha=0.5)
            plt.savefig(f"{output}/AverageStD.png")
            plt.close()

            companygeostdmean = alldata.groupby("company").mean().loc[:, ["ygeostdmean", "ystdmean"]]
            companynorms = pd.merge(alldata, companygeostdmean.reset_index(), on="company").set_index("x")
            normed_geostds = companynorms["ygeostdmean_x"] / companynorms["ygeostdmean_y"]
            normedgroupedgeostds = normed_geostds.groupby(companynorms.index).mean()
            normed_stds = companynorms["ystdmean_x"] / companynorms["ystdmean_y"]
            normedgroupedstds = normed_stds.groupby(companynorms.index).mean()
            lenset = len(normed_geostds) / len(normedgroupedgeostds)
            lencompany = len(normedgroupedgeostds)
            for i in range(int(lenset)):
                plt.plot(
                    normed_geostds.iloc[lencompany*i:lencompany*i+lencompany].index,
                    normed_geostds.iloc[lencompany*i:lencompany*i+lencompany],
                    alpha=0.3
                )
            plt.plot(normedgroupedgeostds.index, normedgroupedgeostds, color="black")
            plt.ylabel("Normalised geometric SD of companies")
            plt.xlabel(weeksafterstr)
            plt.savefig(f"{output}/NormedAverageStD.png")
            plt.close()


            bycop = all_data.loc[all_data["offset"] == 0, :].groupby("COP").mean()
            model = sm.OLS(bycop["diff"] * 100, sm.add_constant(bycop.reset_index(drop=True).index + 1))
            results = model.fit()
            LOBgrad = results.params["x1"]
            LOBFosset = results.params["const"]
            trendline_stats = pd.concat([
                results.params.rename("BestEstimate"),
                results.conf_int().rename(columns={0: "LowerConf", 1: "UpperConf"})
            ], axis=1)
            trendline_stats.to_csv(f"{output}/COPyearDiffTrend.csv")
            plt.rcParams["xtick.minor.visible"] = True
            plt.scatter(bycop.reset_index(drop=True).index + 1, bycop["diff"] * 100, c="black")
            plt.plot(
                [0, len(copdates)],
                [LOBFosset, LOBFosset + len(copdates) * LOBgrad], linestyle="--",
                c="black"
            )
            plt.fill_between([0, copnum], [distofcops[2], distofcops[2]],
                             [distofcops[4], distofcops[4]], alpha=0.2, color="red")
            plt.fill_between([0, copnum], [distofcops[1], distofcops[1]],
                             [distofcops[5], distofcops[5]], alpha=0.2, color="orange")
            plt.fill_between([0, copnum], [distofcops[0], distofcops[0]],
                             [distofcops[6], distofcops[6]], alpha=0.1, color="yellow")
            plt.grid(True, which='minor', alpha=0.2)
            plt.grid(True, which='major', alpha=0.4)
            plt.xlabel(meetingstring)
            plt.ylabel("Mean % change in stock price (%)")
            plt.legend(["Events", "Trend", "25-75%", "10-90%", "5-95%"])
            plt.savefig(f"{output}/ChangeinStockPricePerCOP.png")
            plt.close()

            model = sm.OLS(bycop["geoSD"], sm.add_constant(bycop.reset_index(drop=True).index + 1))
            results = model.fit()
            LOBgrad = results.params["x1"]
            LOBFosset = results.params["const"]
            trendline_stats = pd.concat([
                    results.params.rename("BestEstimate"),
                    results.conf_int().rename(columns={0: "LowerConf", 1: "UpperConf"})
                ], axis=1)
            trendline_stats.to_csv(f"{output}/COPyearVarianceTrend.csv")
            plt.rcParams["xtick.minor.visible"] = True
            plt.scatter(bycop.index + 1, bycop["geoSD"], c="black")
            plt.plot([0, len(copdates)], [LOBFosset, LOBFosset + len(copdates) * LOBgrad],
                     linestyle="--", c="black")
            plt.fill_between([0, copnum], [distofcopsGeoSD[2], distofcopsGeoSD[2]],
                             [distofcopsGeoSD[4], distofcopsGeoSD[4]], alpha=0.2, color="red")
            plt.fill_between([0, copnum], [distofcopsGeoSD[1], distofcopsGeoSD[1]],
                             [distofcopsGeoSD[5], distofcopsGeoSD[5]], alpha=0.2,
                             color="orange")
            plt.fill_between([0, copnum], [distofcopsGeoSD[0], distofcopsGeoSD[0]],
                             [distofcopsGeoSD[6], distofcopsGeoSD[6]], alpha=0.1,
                             color="yellow")
            plt.grid(True, which='minor', alpha=0.2)
            plt.grid(True, which='major', alpha=0.4)
            plt.xlabel(meetingstring)
            plt.ylabel("Mean geoSD of stock price")
            plt.legend(["Trend", "25-75%", "10-90%", "5-95%"])
            plt.savefig(f"{output}/ChangeinGeoSDStockPricePerCOP.png")
            plt.close()

            def copnormfunction(all_data, offset1, distance):
                return all_data.loc[all_data["offset"] == offset1, :].groupby("COP").mean() - (
                        all_data.loc[all_data["offset"] == offset1 - distance, :].groupby(
                            "COP").mean() +
                        all_data.loc[all_data["offset"] == offset1 + distance, :].groupby(
                            "COP").mean()
                ) * 0.5

            linsubcop_length = 3
            distofnormedmeans = pd.concat([
                copnormfunction(all_data, i, linsubcop_length) for i in
                list(range(-maxweek + linsubcop_length, -1 - linsubcop_length)) + list(
                    range(3 + linsubcop_length, maxweek - linsubcop_length + 1))
            ])
            quants_of_normedCOPs = np.nanquantile(distofnormedmeans["diff"],
                                                  [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            quants_of_normedCOPs = [i * 100 for i in quants_of_normedCOPs]
            linsubcop = copnormfunction(all_data, 0, linsubcop_length)
            plt.fill_between([0, copnum], [quants_of_normedCOPs[2], quants_of_normedCOPs[2]],
                             [quants_of_normedCOPs[4], quants_of_normedCOPs[4]], alpha=0.2,
                             color="red")
            plt.fill_between([0, copnum], [quants_of_normedCOPs[1], quants_of_normedCOPs[1]],
                             [quants_of_normedCOPs[5], quants_of_normedCOPs[5]], alpha=0.2,
                             color="orange")
            plt.fill_between([0, copnum], [quants_of_normedCOPs[0], quants_of_normedCOPs[0]],
                             [quants_of_normedCOPs[6], quants_of_normedCOPs[6]], alpha=0.1,
                             color="yellow")
            plt.xlabel(meetingstring)
            plt.plot(linsubcop.index + 1, linsubcop["diff"] * 100)
            plt.ylabel("Mean excess change in stock price (%)")
            plt.legend(["25-75%", "10-90%", "5-95%", "Trend"])
            plt.savefig(f"{output}/NormedChangePerCop.png")
            plt.close()

            quants_of_normedCOPsGeoSD = np.nanquantile(distofnormedmeans["geoSD"],
                                                       [0.05, 0.1, 0.25, 0.5, 0.75, 0.9,
                                                        0.95])
            quants_of_normedCOPsGeoSD = [i * 100 for i in quants_of_normedCOPsGeoSD]

            plt.plot(linsubcop.index + 1, linsubcop["geoSD"] * 100)
            plt.fill_between([0, copnum],
                             [quants_of_normedCOPsGeoSD[2], quants_of_normedCOPsGeoSD[2]],
                             [quants_of_normedCOPsGeoSD[4], quants_of_normedCOPsGeoSD[4]],
                             alpha=0.2, color="red")
            plt.fill_between([0, copnum],
                             [quants_of_normedCOPsGeoSD[1], quants_of_normedCOPsGeoSD[1]],
                             [quants_of_normedCOPsGeoSD[5], quants_of_normedCOPsGeoSD[5]],
                             alpha=0.2, color="orange")
            plt.fill_between([0, copnum],
                             [quants_of_normedCOPsGeoSD[0], quants_of_normedCOPsGeoSD[0]],
                             [quants_of_normedCOPsGeoSD[6], quants_of_normedCOPsGeoSD[6]],
                             alpha=0.1, color="yellow")
            plt.xlabel(meetingstring)
            plt.ylabel("Mean excess geometric SD")
            plt.legend(["Trend", "25-75%", "10-90%", "5-95%"])
            plt.savefig(f"{output}/NormedGeoSDPerCop.png")
            plt.plot(bycop.index + 1, bycop["geoSD"])
            plt.xlabel(meetingstring)
            plt.ylabel("Mean geometric SD of stock price during COP")
            plt.savefig(f"{output}/GeometricSDPerCOP.png")
            plt.close()

            # Plot histograms
            bin_edges = np.histogram_bin_edges(all_data["diff"], bins=50)
            plt.hist(all_data[all_data.offset == 0]["diff"], density=True, bins=bin_edges)
            plt.hist(definitelynotcops["diff"], density=True, bins=bin_edges, alpha=0.6)
            plt.xlabel("Fractional change")
            plt.ylabel("Density of instances")
            plt.legend([copOrOtherLongstring, "Other times"])
            plt.savefig(f"{output}/HistogramOfDiffs.png")
            plt.close()

            binlimithigh = all_data["diff"].mean() + 5 * all_data["diff"].std()
            binlimitlow = all_data["diff"].mean() - 5 * all_data["diff"].std()
            bindf = all_data.loc[
                    (all_data["diff"] < binlimithigh) & (all_data["diff"] > binlimitlow), :]
            bin_edges = np.histogram_bin_edges(bindf["diff"], bins=50)
            plt.hist(bindf.loc[bindf.offset == 0]["diff"], density=True, bins=bin_edges)
            bin_baseline = definitelynotcops.loc[
                (definitelynotcops["diff"] < binlimithigh) & (
                        definitelynotcops["diff"] > binlimitlow), "diff"
            ]
            plt.hist(bin_baseline, density=True, bins=bin_edges, alpha=0.6)
            plt.axvline(
                np.mean(bindf.loc[bindf.offset == 0]["diff"]), color='slateblue', linestyle='-.', linewidth=2,
                label=f"{copOrOtherLongstring} mean"
            )
            plt.axvline(
                np.mean(bin_baseline), color='orange', linestyle='dashed', linewidth=2,
                label="Other times mean"
            )
            plt.xlabel("Fractional change")
            plt.ylabel("Density of instances")
            plt.legend([
                f"{copOrOtherLongstring} mean", "Other times mean",  copOrOtherLongstring, "Other times"
            ])
            plt.savefig(f"{output}/HistogramOfDiffs_trunc.png")

            collected_data = pd.concat([
                pd.DataFrame({"Event": copOrOtherLongstring,
                              "Difference": all_data[all_data.offset == 0]["diff"]}),
                pd.DataFrame({"Event": "No " + copOrOtherLongstring,
                              "Difference": definitelynotcops["diff"]}),
            ])
            sns.violinplot(x=collected_data["Difference"], hue=collected_data["Event"],
                           split=True)
            plt.xlabel("Fractional change")
            plt.ylabel("Density of instances")
            plt.savefig(f"{output}/ViolinOfDiffs.png")
            plt.close()

            bin_edges = np.histogram_bin_edges(all_data["geoSD"], bins=50)
            plt.hist(all_data[all_data.offset == 0]["geoSD"], density=True, bins=bin_edges)
            plt.hist(definitelynotcops["geoSD"], density=True, bins=bin_edges, alpha=0.6)
            plt.xlabel("Geometric standard deviation")
            plt.ylabel("Density of instances")
            plt.legend([copOrOtherLongstring, "Other times times"])
            plt.savefig(f"{output}/HistogramOfGeoSDs.png")
            plt.close()

            histogramdf = all_data
            histlimit = 5 * all_data["geoSD"].std() + all_data["geoSD"].mean()
            histogramdf = histogramdf[histogramdf["geoSD"] < histlimit]
            bin_edges = np.histogram_bin_edges(histogramdf["geoSD"], bins=50)
            plt.hist(histogramdf[histogramdf.offset == 0]["geoSD"], density=True,
                     bins=bin_edges)
            baseline_geosd = definitelynotcops.loc[definitelynotcops["geoSD"] < histlimit, "geoSD"]
            plt.hist(baseline_geosd,
                     density=True, bins=bin_edges, alpha=0.6)
            plt.xlabel("Geometric standard deviation")
            plt.ylabel("Density of instances")
            plt.axvline(
                np.mean(bindf.loc[bindf.offset == 0]["geoSD"]), color='slateblue', linestyle='-.', linewidth=2,
                label=f"{copOrOtherLongstring} mean"
            )
            plt.axvline(
                np.mean(baseline_geosd), color='orange', linestyle='dashed', linewidth=2,
                label="Other times mean"
            )
            plt.legend([
                f"{copOrOtherLongstring} mean", "Other times mean", copOrOtherLongstring, "Other times"
            ])
            plt.savefig(f"{output}/HistogramOfGeoSDs_truncated.png")
            plt.close()

            collected_data = pd.concat([
                pd.DataFrame({"Event": copOrOtherLongstring,
                              "Difference": all_data[all_data.offset == 0]["geoSD"]}),
                pd.DataFrame({"Event": "No " + copOrOtherLongstring,
                              "Difference": definitelynotcops["geoSD"]}),
            ])
            sns.violinplot(x=collected_data["Difference"], hue=collected_data["Event"],
                           split=True)
            plt.xlabel("Geometric standard deviation")
            plt.ylabel("Density of instances")
            plt.savefig(f"{output}/ViolinOfDiffs.png")
            plt.close()

        ### Calculate relative rank and implied statistics

        # define some functions to calculate the probability using only rank information
        def relative_rank(values, target_value, rel=False):
            # Sort the values in ascending order
            sorted_values = sorted(values.append(pd.Series(target_value, index=[0])))
            # Get the relative rank of the target value
            rank = sorted_values.index(target_value) + 1
            if rel:
                rank = rank / (len(values) + 1)
            return rank
        def calc_prob_from_rank(values, target_value):
            # values are the array of observations in general, target value is the putatively extreme value
            rankv = relative_rank(values, target_value)
            # Converts a ranking into a probability using Jenkinson's Formula
            # See Estimating Changing Extremes Using Empirical Ranking Methods
            # Formula is (rank - 0.31) / (N + 0.38).
            return (rankv - 0.31) / (len(values) + 1.38)

        # Construct a dataframe of the results of KS test and student-T test for 2 samples comparing these populations
        # We report the uncorrected data, but also the data after correcting for correlation between observations
        KSdiff = kstest(all_data[all_data.offset==0]["diff"], definitelynotcops["diff"])
        KSgeoSD = kstest(all_data[all_data.offset==0]["geoSD"], definitelynotcops["geoSD"])
        tdiff = ttest_ind(all_data[all_data.offset==0]["diff"], definitelynotcops["diff"])
        tgeoSD = ttest_ind(all_data[all_data.offset==0]["geoSD"], definitelynotcops["geoSD"])
        # Now calculate using the effective number of independent values
        # Calculate the correlation at each timepoint between company data
        n_effective_ratio = (
            all_data.pivot(index=('COP', "offset"), columns="company", values='diff').std() ** 2
        ).sum() / all_data.pivot(index=('COP', "offset"), columns="company", values='diff').cov().sum().sum()
        # Count the number of independent company/time slots * reduced number of companies seen at each time.
        nks1 = np.ceil(all_data[all_data.offset==0].groupby("COP").count()["company"] * n_effective_ratio).sum()
        nks2 = np.ceil(definitelynotcops.groupby(["COP", "offset"]).count()["company"] * n_effective_ratio).sum()
        effective_event_num = int((nks1*nks2)/(nks1+nks2))
        KSdiffcorrel = kstwo.sf(KSdiff.statistic,  effective_event_num)
        KSgeoSDcorrel = kstwo.sf(KSgeoSD.statistic, effective_event_num)

        higeosd = 5*all_data["geoSD"].std() + all_data["geoSD"].mean()
        logeosd = -5*all_data["geoSD"].std() + all_data["geoSD"].mean()
        hidiff = 5*all_data["diff"].std() + all_data["diff"].mean()
        lodiff = -5*all_data["diff"].std() + all_data["diff"].mean()

        truncatedtdiff = ttest_ind(
             [x for x in all_data[all_data.offset==0]["diff"] if (x < hidiff) & (x>lodiff)],
            [x for x in definitelynotcops["diff"] if (x < hidiff) & (x>lodiff)]
        )
        truncatedtgeoSD = ttest_ind(
            [x for x in all_data[all_data.offset==0]["geoSD"] if (x < higeosd) & (x > logeosd)],
            [x for x in definitelynotcops["geoSD"] if (x < higeosd) & (x > logeosd)]
        )
        kstest_results = pd.DataFrame({
            "Test": ["KSDiff", "KSDiffcorrel", "KSgeoSD", "KSgeoSDcorrel", "t-Diff", "t-geoSD",
                     "t-Diff-truncated", "t-geoSD-truncated", "meandiff", "mediandiff"],
            "testStat": [KSdiff[0], KSdiff[0], KSgeoSD[0], KSgeoSD[0], tdiff[0], tgeoSD[0],
                         truncatedtdiff[0], truncatedtgeoSD[0],
                         all_data[all_data.offset==0]["diff"].mean() - definitelynotcops["diff"].mean(),
                         all_data[all_data.offset==0]["diff"].median() - definitelynotcops["diff"].median(),
                        ],
            "pval": [KSdiff[1], KSdiffcorrel, KSgeoSD[1], KSgeoSDcorrel, tdiff[1], tgeoSD[1],
                     truncatedtdiff[1], truncatedtgeoSD[1], 1, 1],
        })
        ranktest = pd.DataFrame({
            "Test": ["RankTestDiff", "RankTestgeoSD"],
            "testStat": [
                relative_rank(
                    definitelynotcops.groupby("offset").mean()["diff"], all_data[all_data.offset==0]["diff"].mean(), rel=True
                ),
                relative_rank(
                    definitelynotcops.groupby("offset").mean()["geoSD"], all_data[all_data.offset==0]["geoSD"].mean(), rel=True
                )
            ],
            "pval": [
                calc_prob_from_rank(definitelynotcops.groupby("offset").mean()["diff"], all_data[all_data.offset==0]["diff"].mean()),
                calc_prob_from_rank(definitelynotcops.groupby("offset").mean()["geoSD"], all_data[all_data.offset==0]["geoSD"].mean()),
            ]
        })
        kstest_results = kstest_results.append(ranktest)
        if norm_group:
            kstest_results = kstest_results.append(pd.DataFrame({
                "Test": ["Normalised"],
                "testStat": [norm_group],
                "pval": [normfrac]
            }))
        kstest_results.to_csv(f"{output}/Kolmogorov_Smirnoff_test_results.csv")
        print(f"Finished analysis for {filetype}")
        print(kstest_results)