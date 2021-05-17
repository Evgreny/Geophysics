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
        ["AG_Depth", "wellName", "S_UCS Предел прочности на сжатие, Мпа", "S_TSTR Предел прочности на растяжение, Мпа",
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

    # if Param == "UCS":
    #     model = RandomForestRegressor(bootstrap=True, max_depth=15, max_features="auto", min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=7) #UCS
    #     model = CatBoostRegressor(depth=5, iterations=200, l2_leaf_reg=1, learning_rate=0.01)
    #     model = xgboost.XGBRegressor(colsample_bytree=0.5, gamma=0.3, learning_rate=0.03, max_depth=3, min_child_weight=13, n_estimators=200, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6) #UCS
    #     model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1) # UCS
    # elif Param == "TSTR":
    #     model = RandomForestRegressor(bootstrap=True, max_depth=11, max_features="auto", min_samples_leaf=5, min_samples_split=5, n_estimators=100, random_state=7)  # TSTR
    #     model = CatBoostRegressor(depth=16, iterations=100, l2_leaf_reg=1, learning_rate=0.1) #TSTR
    #     model = xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.3, learning_rate=0.01, max_depth=4, min_child_weight=0, n_estimators=500, reg_alpha=None, reg_lambda=None, subsample=0.85) #TSTR
    #     model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1) # TSTR
    # elif Param == "FANG":
    #     model = RandomForestRegressor(bootstrap=True, max_depth=11, max_features="auto", min_samples_leaf=1, min_samples_split=6, n_estimators=100, random_state=1)  # FANG
    #     model = CatBoostRegressor(depth=12, iterations=150, learning_rate=0.04) FANG
    #     model = KNeighborsRegressor(n_neighbors=3, n_jobs=-1) # FANG
    # model = xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.01, learning_rate=0.12, max_depth=5, min_child_weight=0, n_estimators=500, reg_alpha=3, reg_lambda=3, subsample=0.85) FANG
    # elif Param == "A_YME":
    #     model = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1)  # A_YME
    #     model = CatBoostRegressor(depth=11, iterations=500, learning_rate=0.02) #A_YME
    #     model = xgboost.XGBRegressor(colsample_bytree=0.4, gamma=0.3, learning_rate=0.1, max_depth=5, min_child_weight=0, n_estimators=150, reg_alpha=0.9, reg_lambda=None, subsample=0.04) # A_YME
    #     model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1) # A_YME
    # elif Param == "A_PR":
    #     model = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1) # A_PR
    #     model = CatBoostRegressor(depth=15, iterations=500, learning_rate=0.01) #A_PR
    #     model = xgboost.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.05, max_depth=4, min_child_weight=30, n_estimators=500, reg_alpha=1e-05, reg_lambda=1e-05, subsample=1.0) # A_PR
    #     model = KNeighborsRegressor(n_neighbors=22, n_jobs=-1) # A_PR
    # elif Param == "M_YME":
    #     model = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="sqrt", min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=1)  # M_YME
    #     model = CatBoostRegressor(depth=4, iterations=500, learning_rate=0.01)  # M_YME
    #     model = xgboost.XGBRegressor(colsample_bytree=0.8, gamma=0.01, learning_rate=0.15, max_depth=4, min_child_weight=0, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=0.75) # M_YME
    #     model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1) # M_YME
    # elif Param == "M_PR":
    #     model = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="auto", min_samples_leaf=1, min_samples_split=4, n_estimators=100, random_state=7)  # M_PR
    #     model = CatBoostRegressor(depth=7, iterations=500, learning_rate=0.01) # M_PR
    #     model = xgboost.XGBRegressor(colsample_bytree=1, gamma=0, learning_rate=0.15, max_depth=6, min_child_weight=0, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=1) # M_PR
    #     model = KNeighborsRegressor(n_neighbors=6, n_jobs=-1) # M_PR
    # model = xgboost.XGBRegressor(colsample_bytree=0.6, gamma=0.01, learning_rate=0.12, max_depth=5, min_child_weight=0, n_estimators=500, reg_alpha=3, reg_lambda=3, subsample=0.85)
    # model = xgboost.XGBRegressor(colsample_bytree=1, gamma=0, learning_rate=0.15, max_depth=6, min_child_weight=0, n_estimators=100, reg_alpha=None, reg_lambda=None, subsample=1)
    # model = xgboost.XGBRegressor()
    # model = CatBoostRegressor()
    # model = LinearRegression(normalize=False, fit_intercept=True)
    # model = ElasticNet(normalize=False, fit_intercept=True)
    model = KNeighborsRegressor(n_neighbors=6, n_jobs=-1)

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

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    Prediction = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        Prediction.append(model.predict(X_test)[0])
    pred = pd.Series(Prediction)
    SST = sum((y - y.mean())**2)
    SSE = sum((pred - y)**2)
    print("LLO", 1-SSE/SST)

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
    Data.reset_index(inplace=True, drop=True)
    plt.hist(Data["AG_Depth"], bins=10)
    plt.show()
    scaler = StandardScaler()
    scale = pd.DataFrame(scaler.fit_transform(Data[["AG_Depth", "G_Общая радиоактивность, API", "G_Содержание естественных радиоактивных элементов, Калий, К, %",
                                                    "G_Содержание естественных радиоактивных элементов, Уран, U, ppm", "G_Содержание естественных радиоактивных элементов, Торий, Th, ppm",
                                                    "G_Объемная плотность, г/см3"]]))
    DatasetShuf = pd.DataFrame({"Depth" : scale[0], "Api" : scale[1], "K" : scale[2], "U" : scale[3], "Th" : scale[4], "VolDen" : scale[2], Param : Data[Param]})

    firstkvart = (DatasetShuf[Param].sort_values(ignore_index=True))[int(DatasetShuf.shape[0] * 1 / 4)]
    thirdkvart = (DatasetShuf[Param].sort_values(ignore_index=True))[int(DatasetShuf.shape[0] * 3 / 4)]
    interkvart = thirdkvart - firstkvart
    blowout =  (DatasetShuf[Param] < (firstkvart - interkvart*1.5)) | (DatasetShuf[Param] > (thirdkvart + interkvart*1.5)) == False
    DatasetShuf= DatasetShuf[blowout]
    DatasetShuf.dropna(inplace=True)
    # # # DatasetShuf.boxplot(column=[Param])
    # # # plt.show()
    DatasetShuf.reset_index(inplace=True, drop=True)
    #
    # # DatasetShuf = Dataset.sample(frac=1, random_state=7)
    X = DatasetShuf.drop(Param, 1)
    y = DatasetShuf[Param]
    # Обучение
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    # model = LinearRegression(normalize=False, fit_intercept=True)
    # model = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="auto", min_samples_leaf=1, min_samples_split=4, n_estimators=100, random_state=7)  # M_PR
    # model = RandomForestRegressor(bootstrap=False, max_depth=10, max_features="sqrt", min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=1)  # M_YME
    # model = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1)  # A_PR
    # model = RandomForestRegressor(bootstrap=False, max_depth=None, max_features="sqrt", min_samples_leaf=4, min_samples_split=3, n_estimators=100, random_state=1)  # A_YME
    # model = RandomForestRegressor(bootstrap=True, max_depth=11, max_features="auto", min_samples_leaf=1, min_samples_split=6, n_estimators=100, random_state=15)  # FANG
    # model = RandomForestRegressor(bootstrap = True, max_depth = 11, max_features = "auto", min_samples_leaf = 5, min_samples_split = 5, n_estimators = 100, random_state=1) #TSTR
    # model = RandomForestRegressor(bootstrap=True, max_depth=15, max_features="auto", min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=7) #UCS


 # Поиск по сетке параметров модели
 #    random_grid = {
 # "bootstrap" : [False, True],
 # 'max_features': ['auto', 'sqrt'],
 # 'max_depth': [5,8,10,12,15,None],
 # 'max_features': ['auto', 'sqrt'],
 # 'min_samples_leaf': [1,2,3,4,5,6],
 # 'min_samples_split': [2,3,4,5,6,7,8],
 # 'n_estimators': [100]}
 #
 #    grid = GridSearchCV(model, random_grid, n_jobs=-1, cv=5)
 #    grid.fit(X,y)
 #    print(grid.best_params_)

    # Prediction = []
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    #     y_train, y_test = y[train_index], y[test_index]
    #     model.fit(X_train, y_train)
    #     Prediction.append(model.predict(X_test)[0])
    # pred = pd.Series(Prediction)
    # SST = sum((y - y.mean())**2)
    # SSE = sum((pred - y)**2)
    # print(1-SSE/SST)

# datas("Dataset.xlsx", "A_YME")
# params = ["UCS", "TSTR", "FANG", "A_YME", "A_PR", "M_YME", "M_PR"]
# for i in params:
#     dataset_former_acoustic("AcousticLearnMech.xlsx", i)

dataset_former_acoustic("AcousticLearnMech.xlsx", "M_PR")
# dataset_former_acoustic("AcousticLearn.xlsx", "FANG")
# dataset_former_acoustic("AcousticLearn.xlsx", "A_YME")
# dataset_former_acoustic("AcousticLearn.xlsx", "A_PR")


# dataset_former_gamma("GammaLearn.xlsx", "UCS")

# print(acoustic_data("AcousticTaskOne.xlsx"))
# gamma_data("FullKernOne.xlsx").to_excel("GammaLearnMech.xlsx", index = False)

# gamma = pd.read_excel("GammaLearn.xlsx")
# print(gamma["A_Краткая литологическая характеристика"])
