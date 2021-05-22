import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
import graphviz
from catboost import CatBoostRegressor
import xgboost

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
        ["AG_Depth", "wellName", "S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
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

    DataLearn = DataLearn.rename(
        columns={"S_UCS Предел прочности на сжатие, Мпа": "UCS", "S_TSTR Предел прочности на растяжение, Мпа": "TSTR",
                 "S_Коэффициент внутреннего трения, tgφ, отн. ед.": "FANG", "A_Динамический Коэфф. Пуассона": "A_PR",
                 "A_Динамический Модуль Юнга, Гпа": "A_YME", "M_Статический Коэфф. Пуассона": "M_PR",
                 "M_Статический Модуль Юнга, ГПа": "M_YME"})

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
        ["AG_Depth", "G_Глубина отбора по ГИС, м", "wellName", "S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
         "S_Коэффициент внутреннего трения, tgφ, отн. ед.", "M_Статический Коэфф. Пуассона",
         "M_Статический Модуль Юнга, ГПа", "A_Динамический Коэфф. Пуассона", "A_Динамический Модуль Юнга, Гпа",
         "G_Общая радиоактивность, API", "G_Содержание естественных радиоактивных элементов, Калий, К, %",
         "G_Содержание естественных радиоактивных элементов, Уран, U, ppm",
         "G_Содержание естественных радиоактивных элементов, Торий, Th, ppm", "G_Объемная плотность, г/см3"]]

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

    DataLearn = DataLearn.rename(
        columns={"S_UCS Предел прочности на сжатие, Мпа": "UCS", "S_TSTR Предел прочности на растяжение, Мпа": "TSTR",
                 "S_Коэффициент внутреннего трения, tgφ, отн. ед.": "FANG", "A_Динамический Коэфф. Пуассона": "A_PR",
                 "A_Динамический Модуль Юнга, Гпа": "A_YME", "M_Статический Коэфф. Пуассона": "M_PR",
                 "M_Статический Модуль Юнга, ГПа": "M_YME"})

    return DataLearn

def litology(name):
    res = name[0] + " " + name[1]
    return res


def datas(FileName, Param):
    print("Old dataset")
    Data = pd.read_excel(FileName)
    Data = Data.rename(
        columns={"S_UCS Предел прочности на сжатие, Мпа": "UCS", "S_TSTR Предел прочности на растяжение, Мпа": "TSTR",
                 "S_Коэффициент внутреннего трения, tgφ, отн. ед.": "FANG", "A_Динамический Коэфф. Пуассона": "A_PR",
                 "A_Динамический Модуль Юнга, Гпа": "A_YME", "M_Статический Коэфф. Пуассона": "M_PR",
                 "M_Статический Модуль Юнга, ГПа": "M_YME"})
    Filter = pd.isna(Data[Param]) == False
    Data = Data[Filter]
    # Обработка столбца литологии, применение Onehotencoding
    Data.dropna(inplace=True, subset=["A_Краткая литологическая характеристика"])
    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.split(" ").apply(litology)

    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.replace(",", "")
    RareLit = ["Переслаивание алевролита", "Тонкое переслаивание", "Алевролит мелко-крупнозернистый",
               "Алевролит глинистый", "Глина аргиллитоподобная", "Аргиллит опесчаненный", "Аргиллит алевритистый"]
    for word in RareLit:
        Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].replace(word, "Rare")
    Data.reset_index(inplace=True, drop=True)

    # Data["A_Краткая литологическая характеристика"].value_counts().plot.barh()
    # plt.show()
    encoder = OneHotEncoder(sparse=False)
    LitEncoder = pd.DataFrame(encoder.fit_transform(Data[["A_Краткая литологическая характеристика"]]))
    Data = Data.join(LitEncoder)
    Data.drop("A_Краткая литологическая характеристика", 1, inplace=True)
    Data.drop_duplicates(subset=Param, inplace=True)
    a = np.array([])
    for i in range(100):
        Data = Data.sample(frac=1, random_state=i)
        Data.reset_index(inplace=True, drop=True)
        X = Data[
            ["A_Пористость, Кп, %", "A_Плотность насыщенной породы, г/см3", "A_Глубина отбора по бурению, м"] + list(
                LitEncoder.columns)]
        y = Data[Param]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        lm = LinearRegression(normalize=True, fit_intercept=True)
        a = np.append(a, cross_val_score(lm, X, y, cv=100, scoring="r2").mean())
        lm.fit(X_train, y_train)
    print(a)
    # Модель линейной регрессии
    # lm = LinearRegression(normalize=True, fit_intercept=True)
    # print("Кросс-валидация линейной модели: " + str(cross_val_score(lm, X, y, cv=5, scoring="r2").mean()))
    # lm.fit(X_train, y_train)
    # print("Отложенная выборка: " + str(lm.score(X_test, y_test)))
    # print(type(cross_val_score(lm, X, y, cv=5, scoring="r2").mean()))

    # Случайный лес
    # rf = RandomForestRegressor(random_state=42, min_samples_leaf=2)
    # print("Кросс-валидация линейной модели: " + str(cross_val_score(rf, X, y, cv=20, scoring="r2").mean()))
    # rf.fit(X_train, y_train)
    # print("Отложенная выборка: " + str(rf.score(X_test, y_test)))
    return Data


def dataset_former_acoustic(FileName, Param):

    Data = pd.read_excel(FileName)
    Filter = pd.isna(Data[Param]) == False
    Data = Data[Filter]
    Data.reset_index(inplace=True, drop=True)

    # Обработка столбца литологии, применение Onehotencoding
    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.split(" ").apply(litology)
    Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].str.replace(",", "")
    RareLit = ["Переслаивание алевролита", "Тонкое переслаивание", "Алевролит мелко-крупнозернистый",
               "Алевролит глинистый", "Глина аргиллитоподобная", "Аргиллит опесчаненный", "Аргиллит алевритистый"]
    for word in RareLit:
        Data["A_Краткая литологическая характеристика"] = Data["A_Краткая литологическая характеристика"].replace(word, "Rare")


    Data.drop_duplicates(subset=Param, inplace=True)
    Data.reset_index(inplace=True, drop=True)

    firstkvart = (Data[Param].sort_values(ignore_index=True))[int(Data.shape[0] * 1 / 4)]
    thirdkvart = (Data[Param].sort_values(ignore_index=True))[int(Data.shape[0] * 3 / 4)]
    interkvart = thirdkvart - firstkvart
    blowout = (Data[Param] < (firstkvart - interkvart * 1.5)) | (
                Data[Param] > (thirdkvart + interkvart * 1.5)) == False
    Data = Data[blowout]
    Data.reset_index(inplace=True, drop=True)

    # if Param == "TSTR" or "FANG":
    #     TSTRFANGdrop = (Data["AG_Depth"] > 900)
    #     Data = Data[TSTRFANGdrop]
    #     Data.reset_index(inplace=True, drop=True)

    # plt.hist(Data["AG_Depth"], bins=10)
    # plt.title(Param)
    # plt.show()
    # Data.boxplot(column=[Param])
    # plt.title(Param)
    # plt.show()

    encoder = OneHotEncoder(sparse=False)
    LitEncoder = pd.DataFrame(encoder.fit_transform(Data[["A_Краткая литологическая характеристика"]]))
    scaler = StandardScaler()
    scale = pd.DataFrame(scaler.fit_transform(Data[["A_Пористость, Кп, %", "A_Плотность насыщенной породы, г/см3", "AG_Depth"]]))
    Dataset = pd.DataFrame({"por" : scale[0], "den" : scale[1], "depth" : scale[2], Param : Data[Param]})
    DatasetShuf = Dataset.join(LitEncoder)
    DatasetShuf.reset_index(inplace=True, drop=True)
    # DatasetShuf = DatasetShuf.sample(frac=1, random_state=9)
    # DatasetShuf.reset_index(inplace=True, drop=True)
    # "Кросс валидация при различном перемешивании"
    # a = np.array([])
    # for i in range(1):
    #     DatasetShuf = DatasetShuf.sample(frac=1, random_state=i)
    #     # Обучение
    #     X = DatasetShuf.drop(Param, 1)
    #     y = DatasetShuf[Param]
    #     # model = LinearRegression(normalize=False, fit_intercept=True)
    #     model = CatBoostRegressor(depth=15, iterations=200, learning_rate=0.01)
    #     cross = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    #     print("Кросс-валидация: " + str(cross))
    #     a = np.append(a, cross)
    # print("Кросс валидация при различном перемешивании", a.mean())

    # Обучение
    # model = LinearRegression(normalize=False, fit_intercept=True)
    X = DatasetShuf.drop(Param, 1)
    y = DatasetShuf[Param]

    if Param == "UCS":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=15, max_features="auto", min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=7) #UCS
        modelCAT = CatBoostRegressor(depth=5, iterations=200, l2_leaf_reg=1, learning_rate=0.01)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.5, gamma=0.3, learning_rate=0.03, max_depth=3, min_child_weight=13, n_estimators=200, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6) #UCS
        modelKNN = KNeighborsRegressor(n_neighbors=5, n_jobs=-1) # UCS
    elif Param == "TSTR":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=11, max_features="auto", min_samples_leaf=5, min_samples_split=5, n_estimators=100, random_state=7)  # TSTR
        modelCAT = CatBoostRegressor(depth=16, iterations=100, l2_leaf_reg=1, learning_rate=0.1) #TSTR
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.3, learning_rate=0.01, max_depth=4, min_child_weight=0, n_estimators=500, reg_alpha=None, reg_lambda=None, subsample=0.85) #TSTR
        modelKNN = KNeighborsRegressor(n_neighbors=5, n_jobs=-1) # TSTR
    elif Param == "FANG":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=11, max_features="auto", min_samples_leaf=1, min_samples_split=6, n_estimators=100, random_state=1)  # FANG
        modelCAT = CatBoostRegressor(depth=12, iterations=150, learning_rate=0.04) # FANG
        modelKNN = KNeighborsRegressor(n_neighbors=3, n_jobs=-1) # FANG
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.01, learning_rate=0.12, max_depth=5, min_child_weight=0, n_estimators=500, reg_alpha=3, reg_lambda=3, subsample=0.85) # FANG
    elif Param == "A_YME":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1)  # A_YME
        modelCAT = CatBoostRegressor(depth=11, iterations=500, learning_rate=0.02) #A_YME
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.4, gamma=0.3, learning_rate=0.1, max_depth=5, min_child_weight=0, n_estimators=150, reg_alpha=0.9, reg_lambda=None, subsample=0.04) # A_YME
        modelKNN = KNeighborsRegressor(n_neighbors=2, n_jobs=-1) # A_YME
    elif Param == "A_PR":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1) # A_PR
        modelCAT = CatBoostRegressor(depth=15, iterations=500, learning_rate=0.01) #A_PR
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.05, max_depth=4, min_child_weight=30, n_estimators=500, reg_alpha=1e-05, reg_lambda=1e-05, subsample=1.0) # A_PR
        modelKNN = KNeighborsRegressor(n_neighbors=22, n_jobs=-1) # A_PR
    elif Param == "M_YME":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="sqrt", min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=1)  # M_YME
        modelCAT = CatBoostRegressor(depth=4, iterations=500, learning_rate=0.01)  # M_YME
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.8, gamma=0.01, learning_rate=0.15, max_depth=4, min_child_weight=0, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=0.75) # M_YME
        modelKNN = KNeighborsRegressor(n_neighbors=2, n_jobs=-1) # M_YME
    elif Param == "M_PR":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="auto", min_samples_leaf=1, min_samples_split=4, n_estimators=100, random_state=7)  # M_PR
        modelCAT = CatBoostRegressor(depth=7, iterations=500, learning_rate=0.01) # M_PR
        modelXGB = xgboost.XGBRegressor(colsample_bytree=1, gamma=0, learning_rate=0.15, max_depth=6, min_child_weight=0, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=1) # M_PR
        modelKNN = KNeighborsRegressor(n_neighbors=6, n_jobs=-1) # M_PR
    modelLR = LinearRegression(normalize=False, fit_intercept=True)
    # modelEL = ElasticNet(normalize=False, fit_intercept=True)

 # Поиск по сетке параметров модели случайного леса
 #    random_grid = {
 # "bootstrap" : [False, True],
 # 'max_features': ['auto', 'sqrt'],
 # 'max_depth': [5,8,10,12,15,None],
 # 'max_features': ['auto', 'sqrt'],
 # 'min_samples_leaf': [1,2,3,4,5,6],
 # 'min_samples_split': [2,3,4,5,6,7,8],
 # 'n_estimators': [100]}

    # catboost_grid = {"depth" : [4,5,8,10,15], "learning_rate" : [0.01, 0.02, 0.03, 0.04, 0.05], "iterations" : [300,500]}
    # xgboost_grid = {
    #    'colsample_bytree':[0.4,0.6,0.8],
    #    'gamma':[0,0.03,0.1,0.3],
    #    'min_child_weight':[0,1,5,15],
    #    'learning_rate':[0.05, 0.1, 0.3],
    #    'max_depth':[3,4,5],
    #    'n_estimators':[200],
    #    'reg_alpha':[1e-5, 0.75, None],
    #    'reg_lambda':[1e-5, 0.45, None],
    #    'subsample':[0.04, 0.6, 0.95, 2]}
    # grid = GridSearchCV(model, xgboost_grid, n_jobs=-1, cv=5)
    # grid.fit(X,y)
    # print(grid.best_params_)
    # {'iterations': 200, 'learning_rate': 0.01, 'max_depth': 5} UCS
    # {'depth': 5, 'iterations': 200, 'l2_leaf_reg': 1, 'learning_rate': 0.01} UCS2
    # {'depth': 15, 'iterations': 500, 'l2_leaf_reg': 1, 'learning_rate': 0.05} TSTR
    # {'depth': 10, 'iterations': 200, 'learning_rate': 0.03} FANG
    # {'depth': 5, 'iterations': 200, 'learning_rate': 0.01} A_YME
    # {'depth': 15, 'iterations': 200, 'learning_rate': 0.01} A_PR
    # {'depth': 5, 'iterations': 300, 'learning_rate': 0.03} M_YME
    # {'depth': 4, 'iterations': 500, 'learning_rate': 0.04} M_YME2
    # {'depth': 4, 'iterations': 500, 'learning_rate': 0.02} M_PR
    # {'colsample_bytree': 0.4, 'gamma': 0.3, 'learning_rate': 0.07, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': 0.45, 'subsample': 0.6} UCS par
    # {'colsample_bytree': 0.6, 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1.5, 'n_estimators': 200, 'reg_alpha': 1e-05, 'reg_lambda': 1e-05, 'subsample': 0.6} TSTR par
    # {'colsample_bytree': 0.6, 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1.5, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': 0.01, 'subsample': 0.95} FANG par
    # {'colsample_bytree': 0.4, 'gamma': 0.03, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': None, 'subsample': 0.04} A_YME par
    # {'colsample_bytree': 0.4, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 15, 'n_estimators': 200, 'reg_alpha': 1e-05, 'reg_lambda': 1e-05, 'subsample': 0.6} A_PR par
    # {'colsample_bytree': 0.8, 'gamma': 0.03, 'learning_rate': 0.3, 'max_depth': 5, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': None, 'reg_lambda': 1e-05, 'subsample': 0.6} M_YME
    # {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': None, 'reg_lambda': None, 'subsample': 0.95} M_PR

    # LeaveOneOut for one model
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    Prediction = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        modelXGB.fit(X_train, y_train)
        pre = modelXGB.predict(X_test)[0]
        print(pre)
        Prediction.append(pre)
    pred = pd.Series(Prediction)
    SST = sum((y - y.mean())**2)
    SSE = sum((pred - y)**2)
    print("LLO", 1-SSE/SST)

    # LeaveOneOut for ensembles
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # modelEL.fit(X_train, y_train)
    #     # exampleEL = modelEL.predict(X_test)[0]
    #
    #     modelRF.fit(X_train, y_train)
    #     exampleRF = modelRF.predict(X_test)[0]
    #
    #     modelXGB.fit(X_train, y_train)
    #     exampleXGB = modelXGB.predict(X_test)[0]
    #     #
    #     modelKNN.fit(X_train, y_train)
    #     exampleKNN = modelKNN.predict(X_test)[0]
    #
    #     modelCAT.fit(X_train, y_train)
    #     exampleCAT = modelCAT.predict(X_test)[0]
    #
    #     Mean = (exampleXGB+exampleCAT+exampleKNN+exampleRF)/4
    #     Prediction.append(Mean)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean()) ** 2)
    # SSE = sum((pred - y) ** 2)
    # print("LLO", 1 - SSE / SST)

    # LeaveOneOut for boosting1
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.4, random_state = 1)
    #     # X_train1.reset_index(inplace=True, drop=True)
    #     # y_train1.reset_index(inplace=True, drop=True)
    #
    #     modelEL.fit(X_train, y_train)
    #     exampleEL = modelEL.predict(X_train)
    #     testEL = modelEL.predict(X_test)
    #
    #     # modelRF.fit(X_train, y_train)
    #     # exampleRF = modelRF.predict(X_test)[0]
    #     #
    #     modelXGB.fit(X_train, y_train)
    #     exampleXGB = modelXGB.predict(X_train)
    #     testXGB = modelXGB.predict(X_test)
    #     # #
    #     # modelKNN.fit(X_train, y_train)
    #     # exampleKNN = modelKNN.predict(X_test)[0]
    #     #
    #     # modelCAT.fit(X_train, y_train)
    #     # exampleCAT = modelCAT.predict(X_test)[0]
    #
    #     train_boost = pd.DataFrame({"EL": exampleEL, "XGB" : exampleXGB, "por" : X_train["por"], "den" : X_train["den"], "depth" : X_train["depth"]})
    #     test_boost = pd.DataFrame({"EL": testEL, "XGB" : testXGB, "por" : X_test["por"], "den" : X_test["den"], "depth" : X_test["depth"]})
    #     modelLR.fit(train_boost, y_train)
    #     pre = modelLR.predict(test_boost)[0]
    #     print(pre)
    #     Prediction.append(pre)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean()) ** 2)
    # SSE = sum((pred - y) ** 2)
    # print("LLO", 1 - SSE / SST)

    # Boosting2
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.4, random_state = 1)
    #     X_train1.reset_index(inplace=True, drop=True)
    #     y_train1.reset_index(inplace=True, drop=True)
    #     X_train2.reset_index(inplace=True, drop=True)
    #     y_train2.reset_index(inplace=True, drop=True)
    #
    #     modelEL.fit(X_train1, y_train1)
    #     exampleEL = modelEL.predict(X_train2)
    #     testEL = modelEL.predict(X_test)
    #
    #     # modelRF.fit(X_train, y_train)
    #     # exampleRF = modelRF.predict(X_test)[0]
    #     #
    #     modelXGB.fit(X_train1, y_train1)
    #     exampleXGB = modelXGB.predict(X_train2)
    #     testXGB = modelXGB.predict(X_test)
    #     # #
    #     # modelKNN.fit(X_train, y_train)
    #     # exampleKNN = modelKNN.predict(X_test)[0]
    #     #
    #     # modelCAT.fit(X_train, y_train)
    #     # exampleCAT = modelCAT.predict(X_test)[0]
    #
    #     # train_boost = pd.DataFrame({"EL": exampleEL, "XGB" : exampleXGB, "por" : X_train2["por"], "den" : X_train2["den"], "depth" : X_train2["depth"]})
    #     # test_boost = pd.DataFrame({"EL": testEL, "XGB" : testXGB, "por" : X_test["por"], "den" : X_test["den"], "depth" : X_test["depth"]})
    #     train_boost = pd.DataFrame({"EL": exampleEL, "XGB" : exampleXGB})
    #     test_boost = pd.DataFrame({"EL": testEL, "XGB" : testXGB})
    #     modelEL.fit(train_boost, y_train2)
    #     pre = modelEL.predict(test_boost)[0]
    #     print(pre)
    #     Prediction.append(pre)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean()) ** 2)
    # SSE = sum((pred - y) ** 2)
    # print("LLO", 1 - SSE / SST)



    # Модель линейной регрессии
    # lm = LinearRegression(normalize=True, fit_intercept=True)
    # print("Кросс-валидация линейной модели: " + str(cross_val_score(lm, X, y, cv=5, scoring="r2").mean()))
    # lm.fit(X_train, y_train)
    # print("Отложенная выборка: " + str(lm.score(X_test, y_test)))

    # Случайный лес
    # rf = RandomForestRegressor(bootstrap=True, max_depth=15, max_features="auto", min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=7)
    # print("Кросс-валидация случайного леса: " + str(cross_val_score(rf, X, y, cv=5, scoring="r2").mean()))
    # rf.fit(X_train, y_train)
    # print("Отложенная выборка: " + str(rf.score(X_test, y_test)))

    # CatBoostRegressor
    # model = CatBoostRegressor(iterations=4, learning_rate=1, depth=4)
    # model.fit(X_train, y_train)
    # print(model.score(X_test, y_test))

    # return Data

def dataset_former_gamma(FileName, Param):

    Data = pd.read_excel(FileName)
    Filter = pd.isna(Data[Param]) == False
    Data = Data[Filter]
    Data.reset_index(inplace=True, drop=True)
    Data.drop_duplicates(subset=Param, inplace=True)
    firstkvart = (Data[Param].sort_values(ignore_index=True))[int(Data.shape[0] * 1 / 4)]
    thirdkvart = (Data[Param].sort_values(ignore_index=True))[int(Data.shape[0] * 3 / 4)]
    interkvart = thirdkvart - firstkvart
    blowout =  (Data[Param] < (firstkvart - interkvart*1.5)) | (Data[Param] > (thirdkvart + interkvart*1.5)) == False
    Dataset = Data[blowout]
    Dataset.reset_index(inplace=True, drop=True)

    # plt.hist(Dataset["G_Глубина отбора по ГИС, м"], bins=10)
    # plt.show()

    scaler = StandardScaler()
    scale = pd.DataFrame(scaler.fit_transform(Dataset[["G_Глубина отбора по ГИС, м", "G_Общая радиоактивность, API", "G_Содержание естественных радиоактивных элементов, Калий, К, %",
                                                    "G_Содержание естественных радиоактивных элементов, Уран, U, ppm", "G_Содержание естественных радиоактивных элементов, Торий, Th, ppm",
                                                    "G_Объемная плотность, г/см3"]]))
    DatasetShuf = pd.DataFrame({"Depth" : scale[0], "Api" : scale[1], "K" : scale[2], "U" : scale[3], "Th" : scale[4], "VolDen" : scale[2], Param : Dataset[Param]})

    # DatasetShuf.boxplot(column=[Param])
    # plt.show()

    if Param == "UCS":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=6, max_features="sqrt", min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=7)  # TSTR
        modelKNN = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.8, gamma=0.3, learning_rate=0.04, max_depth=5, min_child_weight=0, n_estimators=100, reg_alpha=0.85, reg_lambda=0.85, subsample=1)
    elif Param == "TSTR":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=3, min_samples_split=3, n_estimators=500, random_state=1)  # TSTR
        modelKNN = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=1, gamma=1, learning_rate=0.4, max_depth=2, min_child_weight=0, n_estimators=100, reg_alpha=0.85, reg_lambda=0.85, subsample=0.18)
    elif Param == "FANG":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=4, max_features="auto", min_samples_leaf=4, min_samples_split=8, n_estimators=100, random_state=7)  # FANG
        modelKNN = KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0, gamma=0.01, learning_rate=0.1, max_depth=3, min_child_weight=0, n_estimators=100, reg_alpha=0.85, reg_lambda=0.3, subsample=0.55)
    elif Param == "A_YME":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=10, max_features="auto", min_samples_leaf=1, min_samples_split=4, n_estimators=100, random_state=7)  # A_YME
        modelKNN = KNeighborsRegressor(n_neighbors=7, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=1, gamma=0.2, learning_rate=0.1, max_depth=3, min_child_weight=0, n_estimators=100, reg_alpha=0.75, reg_lambda=None, subsample=None)
    elif Param == "A_PR":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=8, max_features="sqrt", min_samples_leaf=6, min_samples_split=8, n_estimators=100, random_state=7) # A_PR
        modelKNN = KNeighborsRegressor(n_neighbors=18, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.9, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=12, n_estimators=50, reg_alpha=None, reg_lambda=None, subsample=None)
    elif Param == "M_YME":
        modelRF = RandomForestRegressor(bootstrap=True, max_depth=6, max_features="auto", min_samples_leaf=1, min_samples_split=5, n_estimators=50, random_state=7)  # M_YME
        modelKNN = KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.8, gamma=0.1, learning_rate=0.06, max_depth=3, min_child_weight=0, n_estimators=100, reg_alpha=1, reg_lambda=0.9, subsample=1)
    elif Param == "M_PR":
        modelRF = RandomForestRegressor(bootstrap=False, max_depth=3, max_features="sqrt", min_samples_leaf=1, min_samples_split=3, n_estimators=50, random_state=7)  # M_PR
        modelKNN = KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
        modelXGB = xgboost.XGBRegressor(colsample_bytree=0.8, gamma=0.5, learning_rate=0.1, max_depth=2, min_child_weight=5, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=1)
    modelLR = LinearRegression(normalize=False, fit_intercept=True)
    modelEL = ElasticNet(normalize=False, fit_intercept=True)

    # # DatasetShuf = Dataset.sample(frac=1, random_state=7)

    X = DatasetShuf.drop(Param, 1)
    y = DatasetShuf[Param]
    # #
    # model = LinearRegression(normalize=False, fit_intercept=True)
    # model = xgboost.XGBRegressor()
    # model = RandomForestRegressor(bootstrap=False, max_depth=3, max_features="sqrt", min_samples_leaf=1, min_samples_split=3, n_estimators=50, random_state=7)
    # model = KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
    # Поиск по сетке параметров модели случайного леса
    # random_grid = {
    #     "bootstrap": [False, True],
    #     'max_features': ['auto', 'sqrt'],
    #     'max_depth': [4, 5, 8, 10, 15, None],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    #     'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
    #     'n_estimators': [200]}
    # {'bootstrap': False, 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 200} UCS
    # {'bootstrap': False, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 200} TSTR
    # {'bootstrap': True, 'max_depth': 4, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200} FANG
    # {'bootstrap': True, 'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 200} A_YME
    # {'bootstrap': True, 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 6, 'min_samples_split': 8, 'n_estimators': 200} A_PR
    # {'bootstrap': True, 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 6, 'min_samples_split': 5, 'n_estimators': 200} M_YME
    # {'bootstrap': False, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 200} M_PR
    # xgboost_grid = {
    #    'colsample_bytree':[0.4,0.6,0.8],
    #    'gamma':[0.01,0.1,0.3],
    #    'min_child_weight':[0,1,5,15],
    #    'learning_rate':[0.1, 0.3],
    #    'max_depth':[3,4,5],
    #    'n_estimators':[100],
    #    'reg_alpha':[1e-5, 0.75, None],
    #    'reg_lambda':[1e-5, 0.45, None],
    #    'subsample':[0.04, 0.6, 0.95]}
    # {'colsample_bytree': 0.8, 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': None, 'reg_lambda': 1e-05, 'subsample': 0.6} UCS
    # {'colsample_bytree': 0.8, 'gamma': 0.01, 'learning_rate': 0.3, 'max_depth': 4, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': 1e-05, 'subsample': 0.95} TSTR
    # {'colsample_bytree': 0.4, 'gamma': 0.01, 'learning_rate': 0.3, 'max_depth': 3, 'min_child_weight': 15, 'n_estimators': 200, 'reg_alpha': None, 'reg_lambda': 1e-05, 'subsample': 0.6} FANG
    # {'colsample_bytree': 0.4, 'gamma': 0.01, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': None, 'subsample': 0.04} M_YME
    # {'colsample_bytree': 0.4, 'gamma': 0.01, 'learning_rate': 0.3, 'max_depth': 3, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 1e-05, 'reg_lambda': 1e-05, 'subsample': 0.04} M_PR
    # {'colsample_bytree': 0.8, 'gamma': 0.01, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 0.75, 'reg_lambda': None, 'subsample': 0.04} A_YME
    # {'colsample_bytree': 0.4, 'gamma': 0.01, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 100, 'reg_alpha': None, 'reg_lambda': 1e-05, 'subsample': 0.04} A_PR
    # model = xgboost.XGBRegressor(colsample_bytree=0.9, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=12, n_estimators=50, reg_alpha=None, reg_lambda=None, subsample=None)
    # model = xgboost.XGBRegressor()
    # grid = GridSearchCV(model, xgboost_grid, n_jobs=-1, cv=5)
    # grid.fit(X, y)
    # print(grid.best_params_)


    # LeaveOneOut for one model
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    #     y_train, y_test = y[train_index], y[test_index]
    #     model.fit(X_train, y_train)
    #     pre = model.predict(X_test)[0]
    #     print(pre)
    #     print(y[test_index[0]])
    #     print("-------")
    #     Prediction.append(pre)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean())**2)
    # SSE = sum((pred - y)**2)
    # print("LLO", 1-SSE/SST)

    # LeaveOneOut for bagging
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    Prediction = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
    # #
        modelLR.fit(X_train, y_train)
        exampleLR = modelLR.predict(X_test)[0]
    # #
    #     modelRF.fit(X_train, y_train)
    #     exampleRF = modelRF.predict(X_test)[0]
    # #
        modelXGB.fit(X_train, y_train)
        exampleXGB = modelXGB.predict(X_test)[0]
    # #     #
        modelKNN.fit(X_train, y_train)
        exampleKNN = modelKNN.predict(X_test)[0]
    # #
    # #     modelCAT.fit(X_train, y_train)
    # #     exampleCAT = modelCAT.predict(X_test)[0]
    # #
        Mean = (exampleKNN+exampleXGB+exampleLR)/3
        print(Mean)
        Prediction.append(Mean)
    pred = pd.Series(Prediction)
    SST = sum((y - y.mean()) ** 2)
    SSE = sum((pred - y) ** 2)
    print("LLO", 1 - SSE / SST)


    # LeaveOneOut for boosting1
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.4, random_state = 1)
    #     # X_train1.reset_index(inplace=True, drop=True)
    #     # y_train1.reset_index(inplace=True, drop=True)
    #
    #     modelLR.fit(X_train, y_train)
    #     exampleLR = modelLR.predict(X_train)
    #     testLR = modelLR.predict(X_test)
    #
    #     # modelRF.fit(X_train, y_train)
    #     # exampleRF = modelRF.predict(X_test)[0]
    #     #
    #     modelXGB.fit(X_train, y_train)
    #     exampleXGB = modelXGB.predict(X_train)
    #     testXGB = modelXGB.predict(X_test)
    #     # #
    #     # modelKNN.fit(X_train, y_train)
    #     # exampleKNN = modelKNN.predict(X_test)[0]
    #     #
    #     # modelCAT.fit(X_train, y_train)
    #     # exampleCAT = modelCAT.predict(X_test)[0]
    #
    #     train_boost = pd.DataFrame({"EL": exampleLR, "XGB" : exampleXGB, "Depth" : X_train["Depth"], "Api" : X_train["Api"], "K" : X_train["K"], "U" : X_train["U"], "Th" : X_train["Th"], "VolDen" : X_train["VolDen"]})
    #     test_boost = pd.DataFrame({"EL": testLR, "XGB" : testXGB, "Depth" : X_test["Depth"], "Api" : X_test["Api"], "K" : X_test["K"], "U" : X_test["U"], "Th" : X_test["Th"], "VolDen" : X_test["VolDen"]})
    #     # train_boost = pd.DataFrame({"RF": exampleRF, "XGB" : exampleXGB})
    #     # test_boost = pd.DataFrame({"RF": testRF, "XGB" : testXGB})
    #     modelRF.fit(train_boost, y_train)
    #     pre = modelRF.predict(test_boost)[0]
    #     print(pre)
    #     Prediction.append(pre)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean()) ** 2)
    # SSE = sum((pred - y) ** 2)
    # print("LLO", 1 - SSE / SST)


    # Boosting2
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.5, random_state = 1)
    #     X_train1.reset_index(inplace=True, drop=True)
    #     y_train1.reset_index(inplace=True, drop=True)
    #     X_train2.reset_index(inplace=True, drop=True)
    #     y_train2.reset_index(inplace=True, drop=True)
    #
    #     # modelEL.fit(X_train1, y_train1)
    #     # exampleEL = modelEL.predict(X_train2)
    #     # testEL = modelEL.predict(X_test)
    #
    #     modelLR.fit(X_train1, y_train1)
    #     exampleLR = modelLR.predict(X_train2)[0]
    #     testLR = modelLR.predict(X_test)
    #
    #     #
    #     modelXGB.fit(X_train1, y_train1)
    #     exampleXGB = modelXGB.predict(X_train2)
    #     testXGB = modelXGB.predict(X_test)
    #     # #
    #     # modelKNN.fit(X_train, y_train)
    #     # exampleKNN = modelKNN.predict(X_test)[0]
    #     #
    #     # modelCAT.fit(X_train, y_train)
    #     # exampleCAT = modelCAT.predict(X_test)[0]
    #     # train_boost = pd.DataFrame({"EL": exampleRF, "XGB" : exampleXGB, "Depth" : X_train2["Depth"], "Api" : X_train2["Api"], "K" : X_train2["K"], "U" : X_train2["U"], "Th" : X_train2["Th"], "VolDen" : X_train2["VolDen"]})
    #     # test_boost = pd.DataFrame({"EL": testRF, "XGB" : testXGB, "Depth" : X_test["Depth"], "Api" : X_test["Api"], "K" : X_test["K"], "U" : X_test["U"], "Th" : X_test["Th"], "VolDen" : X_test["VolDen"]})
    #     train_boost = pd.DataFrame({"RF": exampleLR, "XGB" : exampleXGB})
    #     test_boost = pd.DataFrame({"RF": testLR, "XGB" : testXGB})
    #     modelLR.fit(train_boost, y_train2)
    #     pre = modelLR.predict(test_boost)[0]
    #     print(pre)
    #     Prediction.append(pre)
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean()) ** 2)
    # SSE = sum((pred - y) ** 2)
    # print("LLO", 1 - SSE / SST)


# datas("Dataset.xlsx", "A_YME")
# params = ["UCS", "TSTR", "FANG", "A_YME", "A_PR", "M_YME", "M_PR"]
# for i in params:
#     dataset_former_acoustic("AcousticLearnMech.xlsx", i)

# dataset_former_acoustic("AcousticLearn.xlsx", "A_PR")
# dataset_former_acoustic("AcousticLearn.xlsx", "FANG")
# dataset_former_acoustic("AcousticLearn.xlsx", "A_YME")
# dataset_former_acoustic("AcousticLearn.xlsx", "A_PR")


dataset_former_gamma("GammaLearnNew.xlsx", "TSTR")
# b = dataset_former_gamma("GammaLearnNew.xlsx", "TSTR")
# c = dataset_former_gamma("GammaLearnNew.xlsx", "FANG")
# d = dataset_former_gamma("GammaLearnNew.xlsx", "A_YME")
# e = dataset_former_gamma("GammaLearnNew.xlsx", "A_PR")
# f = dataset_former_gamma("GammaLearnNew.xlsx", "M_YME")
# g = dataset_former_gamma("GammaLearnNew.xlsx", "M_PR")
# print(a, "UCS")
# print(b, "TSTR")
# print(c, "FANG")
# print(d, "A_YME")
# print(e, "A_PR")
# print(f, "M_YME")
# print(g, "M_PR")

# print(acoustic_data("AcousticTaskOne.xlsx"))
# gamma_data("FullKern.xlsx").to_excel("GammaLearnNew.xlsx", index = False)
# gamma = pd.read_excel("GammaLearn.xlsx")
# print(gamma["A_Краткая литологическая характеристика"])
