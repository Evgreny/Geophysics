import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.max_columns", 101)
pd.options.display.max_columns = 999
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 800)

def acoustic_data(FileName):

    print("Данные по акустике в " + FileName)
    FullKern = pd.read_excel(FileName)

    Por = pd.isna(FullKern["A_Пористость, Кп, %"]) == False
    Den = pd.isna(FullKern["A_Плотность насыщенной породы, г/см3"]) == False
    Lit = pd.isna(FullKern["A_Краткая литологическая характеристика"]) == False

    UCS = pd.isna(FullKern["S_UCS Предел прочности на сжатие, Мпа"]) == False
    TSTR = pd.isna(FullKern["S_TSTR Предел прочности на растяжение, Мпа"]) == False
    FANG = pd.isna(FullKern["S_Коэффициент внутреннего трения, tgφ, отн. ед."]) == False
    A_PR = pd.isna(FullKern["A_Динамический Коэфф. Пуассона"]) == False
    A_YME = pd.isna(FullKern["A_Динамический Модуль Юнга, Гпа"]) == False
    M_PR = pd.isna(FullKern["M_Статический Коэфф. Пуассона"]) == False
    M_YME = pd.isna(FullKern["M_Статический Модуль Юнга, ГПа"]) == False

    x = Por & Den & Lit
    y = UCS | TSTR | FANG | A_PR | A_YME | M_PR | M_YME
    filter = x & y
    FirstFilter = FullKern[filter]
    DataLearn = FirstFilter[
        ["wellName", "S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
         "S_Коэффициент внутреннего трения, tgφ, отн. ед.", "M_Статический Коэфф. Пуассона",
         "M_Статический Модуль Юнга, ГПа", "A_Динамический Коэфф. Пуассона", "A_Динамический Модуль Юнга, Гпа",
         "A_Пористость, Кп, %", "A_Плотность насыщенной породы, г/см3", "A_Краткая литологическая характеристика"]]

    print(DataLearn["M_Статический Модуль Юнга, ГПа"].unique())
    print(DataLearn["M_Статический Модуль Юнга, ГПа"].unique().size - 1, "M_YME")

    print(DataLearn["M_Статический Коэфф. Пуассона"].unique())
    print(DataLearn["M_Статический Коэфф. Пуассона"].unique().size - 1, "M_PR")

    print(DataLearn["S_UCS Предел прочности на сжатие, Мпа"].unique())
    print(DataLearn["S_UCS Предел прочности на сжатие, Мпа"].unique().size - 1, "UCS")

    print(DataLearn["S_TSTR Предел прочности на растяжение, Мпа"].unique())
    print(DataLearn["S_TSTR Предел прочности на растяжение, Мпа"].unique().size - 1, "TSTR")

    print(DataLearn["S_Коэффициент внутреннего трения, tgφ, отн. ед."].unique())
    print(DataLearn["S_Коэффициент внутреннего трения, tgφ, отн. ед."].unique().size - 1, "FANG")

    print(DataLearn["A_Динамический Коэфф. Пуассона"].unique())
    print(DataLearn["A_Динамический Коэфф. Пуассона"].unique().size - 1, "A_PR")

    print(DataLearn["A_Динамический Модуль Юнга, Гпа"].unique())
    print(DataLearn["A_Динамический Модуль Юнга, Гпа"].unique().size - 1, "A_YME")

    return DataLearn

def gamma_data(FileName):

    print("Данные по гамме в " + FileName)
    FullKern = pd.read_excel(FileName)

    API = pd.isna(FullKern["G_Общая радиоактивность, API"]) == False
    K = pd.isna(FullKern["G_Содержание естественных радиоактивных элементов, Калий, К, %"]) == False
    U = pd.isna(FullKern["G_Содержание естественных радиоактивных элементов, Уран, U, ppm"]) == False
    Th = pd.isna(FullKern["G_Содержание естественных радиоактивных элементов, Торий, Th, ppm"]) == False
    Vol = pd.isna(FullKern["G_Объемная плотность, г/см3"]) == False

    UCS = pd.isna(FullKern["S_UCS Предел прочности на сжатие, Мпа"]) == False
    TSTR = pd.isna(FullKern["S_TSTR Предел прочности на растяжение, Мпа"]) == False
    FANG = pd.isna(FullKern["S_Коэффициент внутреннего трения, tgφ, отн. ед."]) == False
    A_PR = pd.isna(FullKern["A_Динамический Коэфф. Пуассона"]) == False
    A_YME = pd.isna(FullKern["A_Динамический Модуль Юнга, Гпа"]) == False
    M_PR = pd.isna(FullKern["M_Статический Коэфф. Пуассона"]) == False
    M_YME = pd.isna(FullKern["M_Статический Модуль Юнга, ГПа"]) == False

    x = API & K & U & Th & Vol
    y = UCS | TSTR | FANG | A_PR | A_YME | M_PR | M_YME
    filter = x & y

    FirstFilter = FullKern[filter]
    DataLearn = FirstFilter[
        ["wellName", "S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
         "S_Коэффициент внутреннего трения, tgφ, отн. ед.", "M_Статический Коэфф. Пуассона",
         "M_Статический Модуль Юнга, ГПа", "A_Динамический Коэфф. Пуассона", "A_Динамический Модуль Юнга, Гпа",
         "G_Общая радиоактивность, API", "G_Содержание естественных радиоактивных элементов, Калий, К, %",
         "G_Содержание естественных радиоактивных элементов, Уран, U, ppm", "G_Содержание естественных радиоактивных элементов, Торий, Th, ppm",
         "G_Объемная плотность, г/см3", "G_round"]]

    print(DataLearn["M_Статический Модуль Юнга, ГПа"].unique())
    print(DataLearn["M_Статический Модуль Юнга, ГПа"].unique().size - 1, "M_YME")

    print(DataLearn["M_Статический Коэфф. Пуассона"].unique())
    print(DataLearn["M_Статический Коэфф. Пуассона"].unique().size - 1, "M_PR")

    print(DataLearn["S_UCS Предел прочности на сжатие, Мпа"].unique())
    print(DataLearn["S_UCS Предел прочности на сжатие, Мпа"].unique().size - 1, "UCS")

    print(DataLearn["S_TSTR Предел прочности на растяжение, Мпа"].unique())
    print(DataLearn["S_TSTR Предел прочности на растяжение, Мпа"].unique().size - 1, "TSTR")

    print(DataLearn["S_Коэффициент внутреннего трения, tgφ, отн. ед."].unique())
    print(DataLearn["S_Коэффициент внутреннего трения, tgφ, отн. ед."].unique().size - 1, "FANG")

    print(DataLearn["A_Динамический Коэфф. Пуассона"].unique())
    print(DataLearn["A_Динамический Коэфф. Пуассона"].unique().size - 1, "A_PR")

    print(DataLearn["A_Динамический Модуль Юнга, Гпа"].unique())
    print(DataLearn["A_Динамический Модуль Юнга, Гпа"].unique().size - 1, "A_YME")

    return DataLearn

def litology(name):
    res = name[0] + " " + name[1]
    return res

def dataset_former(FileName):
    Data = pd.read_excel(FileName)
    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.split(" ").apply(litology)
    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.replace(",", "")
    RareLit = ["Переслаивание алевролита", "Тонкое переслаивание", "Алевролит мелко-крупнозернистый",
               "Алевролит глинистый", "Глина аргиллитоподобная", "Аргиллит опесчаненный", "Аргиллит алевритистый"]
    for word in RareLit:
        Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].replace(word, "Rare")
    encoder = OneHotEncoder(sparse=False)

    # Data["A_Краткая литологическая характеристика"].value_counts().plot.barh()
    # plt.show()

    LitEncoder = pd.DataFrame(encoder.fit_transform(Data[["A_Краткая литологическая характеристика"]]))
    Data = Data.join(LitEncoder)
    Data.drop("A_Краткая литологическая характеристика", 1, inplace=True)
    return Data


dataset_former("AcousticLearn.xlsx")