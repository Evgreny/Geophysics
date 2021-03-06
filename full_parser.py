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


def acoustic_gamma_merge(Acoustic, Gamma):
    A_Wells = Acoustic["A_wellName"].unique()
    G_Wells = Gamma["G_wellName"].unique()
    inter = list(set(A_Wells).intersection(set(G_Wells)))
    Adiff = list(set(A_Wells).difference(set(G_Wells)))
    Gdiff = list(set(G_Wells).difference(set(A_Wells)))
    AG_Dataset = pd.DataFrame()
    for well in inter:
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


def acoustgamma_strength_merge(AcoustGamma, Strength):
    AG_Wells = AcoustGamma["wellName"].unique()
    S_Wells = Strength["S_wellName"].unique()
    inter = list(set(AG_Wells).intersection(set(S_Wells)))
    AGdiff = list(set(AG_Wells).difference(set(S_Wells)))
    AGS_Dataset = pd.DataFrame()
    for well in inter:
        AGS_well = AcoustGamma[AcoustGamma["wellName"] == well]
        AGS_well = pd.merge(AGS_well, Strength[Strength["S_wellName"] == well],
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


def acgamstr_mech_merge_combined(AcGamStr, Mech, connect):
    AGS_Wells = AcGamStr["wellName"].unique()
    M_Wells = Mech["M_wellName"].unique()
    inter = list(set(AGS_Wells).intersection(set(M_Wells)))
    AGdiff = list(set(AGS_Wells).difference(set(M_Wells)))
    AGSM_Dataset = pd.DataFrame()
    if connect == "Dec":
        for well in inter:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_well = AGSM_well.assign(G_round=AGSM_well["G_Глубина отбора по ГИС, м"])
            Mech_well = Mech[Mech["M_wellName"] == well]
            Mech_well = Mech_well.assign(M_round=Mech_well["M_Глубина отбора по ГИС, м"])
            AGSM_well = pd.merge(AGSM_well, Mech_well,
                                 how="outer", left_on="G_round", right_on="M_round", )
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
        for well in AGdiff:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
    elif connect == "One":
        for well in inter:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_well = AGSM_well.assign(G_round=AGSM_well["G_Глубина отбора по ГИС, м"].round(0))
            Mech_well = Mech[Mech["M_wellName"] == well]
            Mech_well = Mech_well.assign(M_round=Mech_well["M_Глубина отбора по ГИС, м"].round(0))
            AGSM_well = pd.merge(AGSM_well, Mech_well,
                                 how="outer", left_on="G_round", right_on="M_round", )
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
        for well in AGdiff:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
    elif connect == "Ten":
        for well in inter:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_index = AGSM_well[AGSM_well["G_Глубина отбора по ГИС, м"].isna() == False].index
            AGSM_well = AGSM_well.assign(
                G_round=(AGSM_well.loc[AGSM_index, "G_Глубина отбора по ГИС, м"] / 10).apply(np.floor).astype(int) * 10)
            Mech_well = Mech[Mech["M_wellName"] == well]
            Mech_index = Mech_well[Mech_well["M_Глубина отбора по ГИС, м"].isna() == False].index
            Mech_well = Mech_well.assign(
                M_round=(Mech_well.loc[Mech_index, "M_Глубина отбора по ГИС, м"] / 10).apply(np.floor).astype(int) * 10)
            AGSM_well = pd.merge(AGSM_well, Mech_well,
                                 how="outer", left_on="G_round", right_on="M_round", )
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
        for well in AGdiff:
            AGSM_well = AcGamStr[AcGamStr["wellName"] == well]
            AGSM_Dataset = pd.concat([AGSM_Dataset, AGSM_well])
    AGSM_Dataset.reset_index(inplace=True, drop=True)
    wellNull = AGSM_Dataset.loc[AGSM_Dataset["wellName"].isna()].index
    AGSM_Dataset.loc[wellNull, "wellName"] = AGSM_Dataset.loc[wellNull, "M_wellName"]
    AGSM_Dataset.drop(columns=["M_wellName"], inplace=True)

    return AGSM_Dataset


def full_kern_data(fullsave):

    Acoustic_df = acmechgam_parser.acoustic_full()
    print("Датасет по акустике создан")

    Gamma_df = acmechgam_parser.gamma_full()
    print("Датасет по гамме создан")

    Strength_df = strength_parser.str_full()
    print("Датасет по прочности создан")

    Mech_df = acmechgam_parser.mech_full()
    print("Датасет по механике создан")

    AG_Dataset = acoustic_gamma_merge(Acoustic = Acoustic_df, Gamma = Gamma_df)
    print("Объединение акустики с гаммой прошло успешно")

    AGS_Dataset = acoustgamma_strength_merge(AcoustGamma = AG_Dataset, Strength = Strength_df)
    print("Объединение акустики и гаммы с прочностью прошло успешно")

    AGSM_Dataset = acgamstr_mech_merge_combined(AcGamStr = AGS_Dataset, Mech = Mech_df, connect="Dec")
    AGSM_Dataset.to_excel("FullKern.xlsx", index=False)
    print("Объединение акустики, гаммы и прочности с механикой прошло успешно")

    if fullsave == True:
        Acoustic_df.to_excel("AcousticDataset.xlsx", index=False)
        Gamma_df.to_excel("GammaDataset.xlsx", index=False)
        Strength_df.to_excel("StrengthDataset.xlsx", index=False)
        Mech_df.to_excel("MechDataset.xlsx", index=False)
        AG_Dataset.to_excel("AcoustGammaDataset.xlsx", index=False)
        AGS_Dataset.to_excel("AcoustGammaStrengthDataset.xlsx", index=False)
        print("Промежуточные датасеты сохранены")

def GIS_kern(GIS, FullKern, full = False):
    GIS.drop(0, 0, inplace=True)
    GISwells = GIS.pop("wellName")
    GIS.drop("datasetName", 1, inplace=True)
    GIS = GIS.applymap(lambda x: str(x).replace(",", "."))
    GIS = GIS.astype("float")
    wellnames = pd.read_excel("true_data.xlsx")
    GISwells.replace(list(wellnames["new_name well"]), list(wellnames["old_name well"]), inplace=True)
    GIS["wellName"] = GISwells
    if full == False:
        FullKern = FullKern[["S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
         "S_Коэффициент внутреннего трения, tgφ, отн. ед.", "M_Статический Коэфф. Пуассона",
         "M_Статический Модуль Юнга, ГПа", "A_Динамический Коэфф. Пуассона", "A_Динамический Модуль Юнга, Гпа", "wellName", "G_Глубина отбора по ГИС, м"]]
    GISkern = pd.merge(GIS, FullKern, how = "outer", left_on=["wellName", "DEPT"], right_on=["wellName", "G_Глубина отбора по ГИС, м"])
    return GISkern

# GISdataset = pd.read_csv("GIS.csv", sep=";", na_values=[-9999], low_memory=False)
# kerndataset = pd.read_excel("FullKern.xlsx")
# GIS_kern(GISdataset, kerndataset, full=False).to_csv("GISKern.csv", index=False)

# Data = pd.read_csv("GISKern.csv", low_memory=False)
# print(Data[(pd.isna(Data["S_UCS Предел прочности на сжатие, Мпа"]) == False)])

# full_kern_data(fullsave=True)