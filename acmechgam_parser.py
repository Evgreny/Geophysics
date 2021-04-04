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

# Пути к данным
ROOT_PATH = r"D:\Учеба\Diplom\UsingData"
PATH_TYPE = "ДО"
PATH_DICT = r"D:\Учеба\Diplom\Headers словарь V.6 ОБЩИЙ.xlsx"
TRUE_DATA_PATH = r"D:\Учеба\Diplom\true_data.xlsx"
GIS_DATA_PATH = r"D:\Учеба\Diplom\ГИС готово.csv"

# Словарь наименованйи столбцов для акустики
ACOUSTIC_DICT = pd.read_excel(PATH_DICT, sheet_name="acoustic")
ACOUSTIC_DICT = dict(zip(ACOUSTIC_DICT.Variations, ACOUSTIC_DICT.Unique))

# Словарь наименованйи столбцов для гаммы
GAMMA_DICT = pd.read_excel(PATH_DICT, sheet_name="gamma")
GAMMA_DICT = dict(zip(GAMMA_DICT.Variations, GAMMA_DICT.Unique))

# Допустимые расширения файлов:
EXT_LIST = [".xls", ".xlsx"]

SUBLEVEL_1 = []  # Список путей к папкам месторождений
# Заполняет список путей к папкам месторождений
if PATH_TYPE == "ДО":
    COMPANY = ROOT_PATH.split("\\")[-1]  # имя дочернего общества
    # Перебираем вложенные папки с месторождениями:
    for item in os.listdir(ROOT_PATH):
        cur_path1 = os.path.join(ROOT_PATH, item)
        if os.path.isdir(cur_path1):
            SUBLEVEL_1.append(cur_path1)
elif PATH_TYPE == "Месторождение":
    COMPANY = ROOT_PATH.split("\\")[-2]
    SUBLEVEL_1.append(ROOT_PATH)


SUBLEVEL_2 = []  # Список путей к папкам скважин
# Заполняет список путей к папкам скважин
for folder in SUBLEVEL_1:
    for item in os.listdir(folder):
        cur_path = os.path.join(folder, item)
        if os.path.isdir(cur_path):
            SUBLEVEL_2.append(cur_path)


def file_search(path):
    """Функция принимает путь к папке скважины.
    Возвращает путь к файлам акустики, гаммы и механики в этой папке.
    Файлы ищутся по ключевым словам в названиях файлов.
    Не рассматриваются файлы с датой последнего обновления до 2011 г.
    Если не находит в папке файлов по ключевым словам и дате,
    возвращает значение None."""
    ac_path = None
    g_path = None
    m_path = None
    # Перебираем экселевские файлы в папке в поисках ключевых слов:
    for top, dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1] in EXT_LIST:
                cur_path = os.path.join(top, name)
                if (
                    os.path.isfile(cur_path)
                    and os.path.getmtime(cur_path)
                    > datetime.datetime(2011, 1, 1).timestamp()
                ):
                    if (
                        name.find("куст") > -1
                    ):  # Не учитываем скважину 58
                        ac_path = cur_path
                    elif (
                        name.find("амма") > -1 and name.find("58") == -1
                    ):  # Не учитываем скважину 58
                        g_path = cur_path
                    elif name.find("еханик") > -1 and name.find("~") == -1:
                        m_path = cur_path
    return ac_path, g_path, m_path


def get_data_acg(path):
    """Функция принимает путь к файлу акустики или гаммы.
    Возвращает датафрейм с данными из этого файла.
    Если не удалось извлечь данные, возвращает None."""
    try:
        # Находим не скрытые листы:
        work_book = xlrd.open_workbook(path)
        sheets = work_book.sheets()
        cur_sheet = 0
        for sheet in sheets:
            if sheet.visibility == 0:
                cur_sheet = sheet.name
                break
        # Считываем верх таблицы и ищем название скважины и месторождения:
        data = pd.read_excel(path, sheet_name=cur_sheet, header=None, nrows=15)
        identification = "Unknown"
        for element in data.iloc[:, 0]:
            if isinstance(element, str) and element.find("есторож") > -1:
                identification = element.split(", ")
        # Ищем шапку таблицы:
        data.dropna(thresh=3, inplace=True)  # убираем строки над шапкой
        header_start_ind = data.index[0]
        if len(data[data.iloc[:, 0] == 1]) > 0:
            data = data[data.iloc[:, 0] == 1]
            header_end_ind = data.index[0] - 1
        else:  # Если в первом столбце только текстовые значения
            data.iloc[:, 1] = pd.to_numeric(data.iloc[:, 1], errors="coerce")
            data = data.dropna()
            header_end_ind = data.index[0] - 1
        # Считываем всю таблицу, указывая строки расположения заголовков:
        indexes = [i for i in range(header_start_ind, header_end_ind + 1)]
        data = pd.read_excel(path, sheet_name=cur_sheet, header=indexes)

        if path == r"D:\Учеба\12 сем\UsingData\Me-1\58по\МЕХАНИКА\Акустика_65по.xlsx":

            data = data.iloc[:,:22]

        # Добавляем в таблицу столбцы с названием месторождения и скважины:
        if len(identification) == 2:
            data["Месторождение"] = identification[0]
            data["Скважина"] = identification[1]
        else:
            data["Месторождение"] = "".join(identification)
        data["ДО"] = COMPANY  # добавляем название дочернего общества
        return data
    except Exception as error:
        print(f"В файле {path}")
        print("Ошибка при вызове функции get_data_acg():", error)
        return None


def get_data_mech(path):
    """Функция принимает путь к файлу механики.
    Возвращает датафрейм с данными из этого файла.
    Если не удалось извлечь данные, возвращает None."""
    try:
        cols = [
            "Месторождение",
            "Пласт",
            "Глубина отбора по ГИС, м",
            "Литология",
            "Номер образца",
            "Пористость, %",
            "Газопроницаемость, мД",
            "Плотность скелета, г/см3",
            "Статический Модуль Юнга, ГПа",
            "Коэффициент сжимаемости твердых частиц (β), ГПа-1",
            "Статический Коэфф. Пуассона",
        ]
        m_df = pd.DataFrame(columns=cols)
        wb = xlrd.open_workbook(filename=path)
        sheets_list = wb.sheet_names()
        for label in sheets_list:
            sheet = wb.sheet_by_name(label)
            if sheet.cell_value(0, 0) == "Привязка керна":
            #     continue
            # else:
                for i in [9, 10, 12, 15]:
                    if (
                        i <= sheet.nrows - 2
                        and sheet.cell_value(i, 0) == "Модуль Юнга (Ест), ГПа"
                    ):
                        new_row = dict()
                        mech_df = pd.read_excel(path, sheet_name=label, header=None)
                        # Выбираем строки, содержащие таблицы с 5 и 4 столбцами:
                        mech_df.dropna(thresh=4, inplace=True)
                        # Извлекаем значения из таблицы с 5 столбцами:
                        new_row["Месторождение"] = mech_df.iloc[1, 0]
                        new_row["Пласт"] = mech_df.iloc[1, 2]
                        new_row["Глубина отбора по ГИС, м"] = mech_df.iloc[1, 3]
                        new_row["Литология"] = mech_df.iloc[1, 5]
                        # Извлекаем значения из таблицы с 4 столбцами:
                        new_row["Номер образца"] = mech_df.iloc[3, 0]
                        new_row["Пористость, %"] = mech_df.iloc[3, 1]
                        new_row["Газопроницаемость, мД"] = mech_df.iloc[3, 3]
                        new_row["Плотность скелета, г/см3"] = mech_df.iloc[3, 5]
                        # Извлекаем значения из вертикальной таблицы:
                        new_row["Статический Модуль Юнга, ГПа"] = sheet.cell_value(i, 5)
                        new_row[
                            "Коэффициент сжимаемости твердых частиц (β), ГПа-1"
                        ] = sheet.cell_value(i + 1, 5)
                        new_row["Статический Коэфф. Пуассона"] = sheet.cell_value(
                            i + 2, 5
                        )
                        m_df = m_df.append(new_row, ignore_index=True)
        m_df["mechanic_path"] = path
        m_df["mechanic_file"] = os.path.basename(path)
        if len(m_df) > 0:
            m_df = m_df.add_prefix("M_")
            return m_df
        else:
            print(f"Не удалось сформировать датасет по механике из файла \n{path}")
            return None
    except Exception as error:
        print(f"В файле {path}")
        print("Ошибка при вызове функции get_data_mech():", error)
        return None


def rename_headers(data_frame, df_type, path):
    """Функция принимает датафрейм, тип данных ('acoustic', 'gamma') и путь к файлу.
    Возвращает датафрейм с переименованными заголовками.
    Если не удалось переименовать заголовки, возвращает None."""
    try:
        if df_type == "acoustic":
            dictionary = ACOUSTIC_DICT
        elif df_type == "gamma":
            dictionary = GAMMA_DICT

        headers = list(data_frame.columns)

        if isinstance(headers[0], str):  # Если заголовок изначально одноуровневый
            data_frame.columns = data_frame.columns.str.strip().str.replace("  ", " ").replace("\n", "")
            data_frame = data_frame.rename(dictionary, axis="columns")

        elif isinstance(headers[0], tuple):  # Если многоуровневый
            oneline_header = []
            for column in headers:
                oneline_column = []
                for subheader in column:
                    if subheader.find("Unnamed") == -1:
                        oneline_column.append(subheader.strip())
                oneline_column = (
                    " ".join(oneline_column).replace("  ", " ").replace("\n", "")
                )
                len_of_columns = len(oneline_column) // 2
                if oneline_column[:len_of_columns] == oneline_column[len_of_columns:]:
                    oneline_column = oneline_column[:len_of_columns]
                oneline_header.append(oneline_column)
            data_frame.columns = oneline_header  # преобразуем заголовок в однострочный
            data_frame = data_frame.rename(dictionary, axis="columns")  # переименовыем

        # Добавляем столбцы с путем к исходному файлу и его названием:
        data_frame[df_type + "_path"] = path
        data_frame[df_type + "_file"] = os.path.basename(path)

        # Добавляем к заголовкам префиксы, чтобы можно было различать столбцы
        # из разных таблиц при последующем объединении в общий датасет:
        if df_type == "acoustic":
            data_frame = data_frame.add_prefix("A_")
        elif df_type == "gamma":
            data_frame = data_frame.add_prefix("G_")

        # Если в таблице есть столбцы с одинаковыми названиями, удаляем дубликаты:
        data_frame = data_frame.loc[:, ~data_frame.columns.duplicated()]

        return data_frame

    except Exception as error:
        print(f"В файле {path}")
        print("Ошибка при вызове функции rename_headers():", error)
        return None


def clean_data(dataframe, path):
    """Функция принимает датафрейм и путь к файлу по акустике или гамме,
    удаляет из него строки с числовой нумерацией столбцов и "Продолжение...".
    Возвращает транформированный датафрейм.
    Если не удалось обработать данные, возвращает None."""
    try:
        dataframe = dataframe.iloc[1:, :]  # Удаляем первую строку после шапки
        # в которой идет числовая нумерация столбцов

        # Удаляем строки внутри таблицы, в которых идет числовая нумерация,
        # и предшествующие им пустые строки со словами "Продолжение", "Окончание" и др.
        util_col = dataframe.columns[7]
        del_rows = dataframe.loc[dataframe[util_col].isin([5, 6, 7, 8]), util_col].index
        index = []  # Собираем в список индексы строк, которые нужно удалить
        for ind in del_rows:
            index.append(ind - 1)
            index.append(ind)
        dataframe = dataframe.drop(index)
        # Удаляем полностью пустые строки:
        dataframe.dropna(thresh=6, inplace=True)
        dataframe = dataframe.reset_index(drop=True)
        return dataframe

    except Exception as error:
        print(f"В файле {path}")
        print("Ошибка при вызове функции clean_data():", error)
        return None


def acoustic_full():
    """Функция принимает путь к папке месторождения.
    Возвращает объединенный датафрейм файлов акустики."""
    DatasetAcoustic = pd.DataFrame()
    for skv in SUBLEVEL_2:
        try:
            ac_path, g_path, m_path = file_search(skv)
            acoustics_df = get_data_acg(ac_path)
            acoustics_df = rename_headers(acoustics_df, "acoustic", ac_path)
            acoustics_df = clean_data(acoustics_df, ac_path)
            acoustics_df["A_wellName"] = skv.split("\\")[-1]
            DatasetAcoustic = pd.concat([DatasetAcoustic, acoustics_df])
            print(skv+" данные по акустике добавлены")
        except Exception as error:
            print(f"В папке {skv} аккустика не взялась")
    DatasetAcoustic.replace("52A", "52р", inplace=True)
    DatasetAcoustic.replace("111A", "111по", inplace=True)
    DatasetAcoustic["A_Глубина отбора по бурению, м"] = (
            DatasetAcoustic["A_Интервал отбора кровля, м"] + DatasetAcoustic["A_Место взятия от верха, м"])
    DatasetAcoustic["A_Глубина отбора по бурению, м"] = DatasetAcoustic["A_Глубина отбора по бурению, м"].round(1)
    return DatasetAcoustic


def gamma_full():
    """Функция принимает путь к папке месторождения.
    Возвращает объединенный датафрейм файлов гаммы."""
    DatasetGamma = pd.DataFrame()
    for skv in SUBLEVEL_2:
        try:
            ac_path, g_path, m_path = file_search(skv)
            gamma_df = get_data_acg(g_path)
            gamma_df = rename_headers(gamma_df, "gamma", g_path)
            gamma_df = clean_data(gamma_df, g_path)
            wellName = skv.split("\\")[-1]
            gamma_df["G_wellName"] = wellName
            try:
                # if wellName == "52р":
                if wellName in ["52р", "60", "61", "65по", "65G"]:

                    gamma_df["G_Глубина отбора по ГИС, м"] = (
                    gamma_df["G_Интервал отбора керна после привязки кровля, м"]
                    + gamma_df["G_Глубина отбора по ГИС, м"])

                    if wellName == "52р":
                        gamma_df["G_Глубина отбора по бурению, м"] = (
                                gamma_df["G_Интервал отбора керна до привязки кровля, м"]
                                + gamma_df["G_Вынос керна, м"])
                    else:
                        gamma_df["G_Глубина отбора по бурению, м"] = np.nan

                    print(f"----------------------------------В файле {g_path} исправлена глубина по бурению и по ГИС.")
            except Exception as error:
                print(f"----------------------------------В файле {g_path} ошибка по глубине")

            DatasetGamma = pd.concat([DatasetGamma, gamma_df])
            print(skv+" данные по гамме добавлены")
        except Exception as error:
            print(f"В папке {skv} гамма не взялась")
    DatasetGamma.replace("65G", "65по", inplace=True)

    DatasetGamma["G_Глубина отбора по бурению, м"] = DatasetGamma["G_Глубина отбора по бурению, м"].round(1)
    DatasetGamma["G_Глубина отбора по ГИС, м"] = DatasetGamma["G_Глубина отбора по ГИС, м"].round(1)
    return DatasetGamma


def mech_full():
    """Функция принимает путь к папке месторождения.
    Возвращает объединенный датафрейм файлов механики."""
    DatasetMech = pd.DataFrame()
    for skv in SUBLEVEL_2:
        try:
            ac_path, g_path, m_path = file_search(skv)
            if "65по" in m_path:
                mech_df = pd.read_excel(m_path)
            else:
                mech_df = get_data_mech(m_path)
            mech_df["M_wellName"] = skv.split("\\")[-1]
            DatasetMech = pd.concat([DatasetMech, mech_df])
            print(skv+" данные по механике добавлены")
        except Exception as error:
            print(f"В папке {skv} механика не взялась")

    DatasetMech["M_Глубина отбора по ГИС, м"] = DatasetMech["M_Глубина отбора по ГИС, м"].round(1)
    DatasetMech.reset_index(inplace=True, drop=True)
    return DatasetMech