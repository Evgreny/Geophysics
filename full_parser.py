import os
import datetime
import copy
import xlrd
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 101)
pd.options.display.max_columns = 999
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 800)


Acoustic = pd.read_excel("AcousticDataset.xlsx")
Gamma = pd.read_excel("GammaDataset.xlsx")
A_Wells = Acoustic["A_wellName"].unique()
G_Wells = Gamma["G_wellName"].unique()
inter = list(set(A_Wells).intersection(set(G_Wells)))
Adiff = list(set(A_Wells).difference(set(G_Wells)))
print(Adiff)
Gdiff = list(set(G_Wells).difference(set(A_Wells)))
print(Gdiff)
AG_Dataset = pd.DataFrame()
for well in inter:
    print(well)
    if well not in ["60" , "61", "65по"]:
        AG_well = pd.DataFrame()
        G_Depth = Gamma[Gamma["G_wellName"] == well]["G_Глубина отбора по бурению, м"]
        A_Depth = Acoustic[Acoustic["A_wellName"] == well]["A_Глубина отбора по бурению, м"]
        G_DepthDiff = list(set(A_Depth).difference(set(G_Depth)))
        if len(G_DepthDiff) != 0:
            G_DepthDiff = pd.Series(G_DepthDiff)
            AG_well["AG_Depth"] = G_Depth.append(G_DepthDiff).sort_values()
        else:
            AG_well["AG_Depth"] = pd.Series(G_Depth)


        AG_well = pd.merge(AG_well, Acoustic[Acoustic["A_wellName"] == well],
            how="outer", left_on="AG_Depth", right_on="A_Глубина отбора по бурению, м",)
        AG_Dataset = pd.concat([AG_Dataset, AG_well])

    else:
        AG_well = Acoustic[Acoustic["A_wellName"] == well]
        AG_well = AG_well.assign(AG_Depth=AG_well["A_Глубина отбора по бурению, м"])
        AG_Dataset = pd.concat([AG_Dataset, AG_well])

for well in Adiff:
     AG_well = Acoustic[Acoustic["A_wellName"] == well]
     AG_well = AG_well.assign(AG_Depth = AG_well["A_Глубина отбора по бурению, м"])
     AG_Dataset = pd.concat([AG_Dataset, AG_well])

# for well in Gdiff:
#     AG_well = Gamma[Gamma["G_wellName"] == well]
#     AG_well["AG_Depth"] = AG_well["G_Глубина отбора по бурению, м"]
#     AG_Dataset = pd.concat([AG_Dataset, AG_well])

AG_Dataset.to_excel("AG.xlsx", index = False)

AG = pd.read_excel("AG.xlsx")
print(AG.groupby("A_wellName").describe())
print(AG.shape)