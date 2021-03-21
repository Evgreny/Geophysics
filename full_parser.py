import os
import datetime
import copy
import xlrd
import numpy as np
import pandas as pd
import acmechgam_parser
import strength_parser


pd.set_option("display.max_columns", 101)
pd.options.display.max_columns = 999
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 800)


def acoustic_gamma_merge():
    Acoustic = pd.read_excel("AcousticDataset.xlsx")
    Gamma = pd.read_excel("GammaDataset.xlsx")
    A_Wells = Acoustic["A_wellName"].unique()
    G_Wells = Gamma["G_wellName"].unique()
    inter = list(set(A_Wells).intersection(set(G_Wells)))
    Adiff = list(set(A_Wells).difference(set(G_Wells)))
    Gdiff = list(set(G_Wells).difference(set(A_Wells)))
    AG_Dataset = pd.DataFrame()
    for well in inter:
        print(well)
        if well not in ["60", "61", "65по"]:
            AG_well = pd.DataFrame()
            G_Depth = Gamma[Gamma["G_wellName"] == well]["G_Глубина отбора по бурению, м"]
            A_Depth = Acoustic[Acoustic["A_wellName"] == well]["A_Глубина отбора по бурению, м"]
            G_DepthDiff = list(set(A_Depth).difference(set(G_Depth)))
            if well == "52р":
                AG_well = Gamma[Gamma["G_wellName"] == well]
                AG_well = AG_well.assign(AG_Depth=AG_well["G_Глубина отбора по бурению, м"])
                AG_well = pd.merge(AG_well, Acoustic[Acoustic["A_wellName"] == well],
                                   how="outer", left_on="AG_Depth", right_on="A_Глубина отбора по бурению, м", )
                ind = AG_well.loc[AG_well["AG_Depth"].isna()].index
                AG_well.loc[ind, "AG_Depth"] = AG_well.loc[ind, "A_Глубина отбора по бурению, м"]
                AG_well = AG_well.sort_values(by=["AG_Depth"])
            else:
                if len(G_DepthDiff) != 0:
                    G_DepthDiff = pd.Series(G_DepthDiff)
                    AG_well["AG_Depth"] = G_Depth.append(G_DepthDiff).sort_values()
                else:
                    AG_well["AG_Depth"] = pd.Series(G_Depth)

                AG_well = pd.merge(AG_well, Acoustic[Acoustic["A_wellName"] == well],
                                   how="outer", left_on="AG_Depth", right_on="A_Глубина отбора по бурению, м", )
                AG_well = pd.merge(AG_well, Gamma[Gamma["G_wellName"] == well],
                                   how="outer", left_on="AG_Depth", right_on="G_Глубина отбора по бурению, м", )
            AG_Dataset = pd.concat([AG_Dataset, AG_well])

        else:
            AC_well = Acoustic[Acoustic["A_wellName"] == well]
            AC_well = AC_well.assign(AG_Depth=AC_well["A_Глубина отбора по бурению, м"])
            G_well = Gamma[Gamma["G_wellName"] == well]
            G_well = G_well.assign(AG_Depth=G_well["G_Глубина отбора по бурению, м"])
            AG_Dataset = pd.concat([AG_Dataset, AC_well])
            AG_Dataset = pd.concat([AG_Dataset, G_well])

    for well in Adiff:
        AG_well = Acoustic[Acoustic["A_wellName"] == well]
        AG_well = AG_well.assign(AG_Depth=AG_well["A_Глубина отбора по бурению, м"])
        AG_Dataset = pd.concat([AG_Dataset, AG_well])

    for well in Gdiff:
        AG_well = Gamma[Gamma["G_wellName"] == well]
        AG_well = AG_well.assign(AG_Depth=AG_well["G_Глубина отбора по бурению, м"])
        AG_Dataset = pd.concat([AG_Dataset, AG_well])

    AG_Dataset.drop_duplicates(inplace=True)
    AG_Dataset.reset_index(inplace=True, drop=True)
    AG_Dataset = AG_Dataset.assign(wellName=AG_Dataset["G_wellName"])
    wellNull = AG_Dataset.loc[AG_Dataset["wellName"].isna()].index
    AG_Dataset.loc[wellNull, "wellName"] = AG_Dataset.loc[wellNull, "A_wellName"]
    AG_Dataset.drop(columns = ["G_wellName", "A_wellName"], inplace=True)

    return AG_Dataset


def acoustgamma_strength_merge():
    AcoustGamma = pd.read_excel("AcoustGammaDataset.xlsx")
    Strenght = pd.read_excel("StrengthDataset.xlsx")
    AG_Wells = AcoustGamma["wellName"].unique()
    S_Wells = Strenght["S_wellName"].unique()
    inter = list(set(AG_Wells).intersection(set(S_Wells)))
    AGdiff = list(set(AG_Wells).difference(set(S_Wells)))
    AGS_Dataset = pd.DataFrame()
    for well in inter:
        AGS_well = AcoustGamma[AcoustGamma["wellName"] == well]
        AGS_well = pd.merge(AGS_well, Strenght[Strenght["S_wellName"] == well],
                            how="outer", left_on="AG_Depth", right_on="S_Глубина отбора по бурению, м", )
        AGS_Dataset = pd.concat([AGS_Dataset, AGS_well])
    for well in AGdiff:
        AGS_well = AcoustGamma[AcoustGamma["wellName"] == well]
        AGS_Dataset = pd.concat([AGS_Dataset, AGS_well])
    AGS_Dataset.reset_index(inplace=True, drop=True)
    wellNull = AGS_Dataset.loc[AGS_Dataset["wellName"].isna()].index
    AGS_Dataset.loc[wellNull, "wellName"] = AGS_Dataset.loc[wellNull, "S_wellName"]
    AGS_Dataset.drop(columns=["S_wellName"], inplace=True)

    return AGS_Dataset


def acgamstr_mech_merge():
    AcGamStr = pd.read_excel("AcoustGammaStrengthDataset.xlsx")
    Mech = pd.read_excel("MechDataset.xlsx")
    AGS_Wells = AcGamStr["wellName"].unique()
    M_Wells = Mech["M_wellName"].unique()
    inter = list(set(AGS_Wells).intersection(set(M_Wells)))
    AGdiff = list(set(AGS_Wells).difference(set(M_Wells)))
    AGSM_Dataset = pd.DataFrame()
    for well in inter:
        AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
        AGSM_well = pd.merge(AGSM_well, Mech[Mech["M_wellName"] == well],
                            how="outer", left_on="G_Глубина отбора по ГИС, м", right_on="M_Глубина отбора по ГИС, м", )
        AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
    for well in AGdiff:
        AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
        AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
    AGSM_Dataset.reset_index(inplace=True, drop=True)
    wellNull = AGSM_Dataset.loc[AGSM_Dataset["wellName"].isna()].index
    AGSM_Dataset.loc[wellNull, "wellName"] = AGSM_Dataset.loc[wellNull, "M_wellName"]
    AGSM_Dataset.drop(columns=["M_wellName"], inplace=True)

    return AGSM_Dataset


def full_kern_data():
    Acoustic = acmechgam_parser.acoustic_full()
    Acoustic.to_excel("AcousticDataset.xlsx", index=False)
    print("Датасет по акустике создан")

    Gamma = acmechgam_parser.gamma_full()
    Gamma.to_excel("GammaDataset.xlsx", index=False)
    print("Датасет по гамме создан")

    Strength = strength_parser.str_full()
    Strength.to_excel("StrengthDataset.xlsx", index=False)
    print("Датасет по прочности создан")

    Mech = acmechgam_parser.mech_full()
    Mech.to_excel("MechDataset.xlsx", index=False)
    print("Датасет по механике создан")

    AG_Dataset = acoustic_gamma_merge()
    AG_Dataset.to_excel("AcoustGammaDataset.xlsx", index=False)
    print("Объединение акустики с гаммой прошло успешно")

    AGS_Dataset = acoustgamma_strength_merge()
    AGS_Dataset.to_excel("AcoustGammaStrengthDataset.xlsx", index=False)
    print("Объединение акустики и гаммы с прочностью прошло успешно")

    AGSM_Dataset = acgamstr_mech_merge()
    AGSM_Dataset.to_excel("FullKern.xlsx", index=False)
    print("Объединение акустики, гаммы и прочности с механикой прошло успешно")