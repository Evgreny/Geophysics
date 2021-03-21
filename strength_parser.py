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


def file_search_st(path):
    """Функция принимает путь к папке скважины.
    Возвращает путь к файлам прочности в этой папке.
    Файлы ищутся по ключевым словам в названиях файлов.
    Не рассматриваются файлы с датой последнего обновления до 2011 г.
    Если не находит в папке файлов по ключевым словам и дате,
    возвращает значение None."""
    # Перебираем экселевские файлы в папке в поисках ключевых слов:
    strength_pathlist = []
    ext_list = [".xls", ".xlsx"]
    for top, dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1] in ext_list:
                the_path = os.path.join(top, name)
                if os.path.isfile(the_path):
                    if name.find("жати") > -1:
                        strength_pathlist.append(the_path)
                    elif name.find("стяже") > -1:
                        strength_pathlist.append(the_path)
                    elif name.find("рочно") > -1 and name.find("аспорт") == -1:
                        strength_pathlist.append(the_path)

    for name in strength_pathlist:
        if name.find("~$") > -1:
            strength_pathlist.remove(name)

    if len(strength_pathlist) == 0:
        print("\tНе найдены данные по прочности.")
        return None

    return strength_pathlist


def join_headers(headers, experiment_type, duration_type):
    """Функция принимает список кортежей из подзаголовков,
    возвращает объединенную строку, из которой удалены
    подзаголовки 'Unnamed', добавлено указание на тип
    исследования и тип сжатия для столбца 'предел прочности'."""

    for i in range(len(headers)):
        # Преобразуем кортеж в список:
        string = list(headers[i])

        # Удаляем подзаголовки без названий:
        if string[-1].find("Unnamed") > -1:
            del string[-1]
        if len(string) > 1 and string[1].find("Unnamed") > -1:
            del string[1]

        # Преобразуем список в строку без разделителей:
        string = ("").join(string)
        string = string.strip().replace("  ", " ")

        # Проверяем, что текст заголовка по повторяется:
        len_string = len(string) // 2
        if string[:len_string] == string[len_string:]:
            string = string[:len_string]

        # Если это заголовок "предела прочности":
        if string.find("редел") > -1:
            # Добавляем указние на тип исследования и тип прочности:
            if experiment_type:
                string += "_" + experiment_type
            if duration_type:
                string += "_" + duration_type

        # Обновляем элемент списка:
        headers[i] = string

    # Возвращаем список строк:
    return headers


def get_data_st(path):
    """Функция принимает датафрейм.
    Возвращает датафрейм с данными из этого файла.
    Если не удалось извлечь данные, возвращает None."""
    try:
        experiment_type = []
        load_type = []
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
                if identification[0].find("\n"):
                    identnew = identification[0].split("\n")
                    for eli in identnew:
                        if eli.find("есторож") > -1:
                            identification[0] = eli
        #                             print('eli=',eli)

        if identification == "Unknown":
            try:
                for element in data.iloc[:, 1]:
                    if isinstance(element, str) and element.find("есторож") > -1:
                        identification = element.split(", ")
            except:
                print("error")

        for element in data.iloc[:, 0]:
            if isinstance(element, str):
                if element.find("сесторон") > -1:
                    experiment_type = "всестороние"
                    break
                if element.find("дноосн") > -1:
                    experiment_type = "одноосное"
                    break
                else:
                    experiment_type = "no type experiment"

        for element in data.iloc[:, 0]:
            if isinstance(element, str):
                if element.find("жати") > -1 and element.find("стяж") == -1:
                    load_type = "сжатие"
                    break
                if element.find("стяж") > -1 and element.find("жати") == -1:
                    load_type = "растяжение"
                    break
                if element.find("стяж") > -1 and element.find("жати") > -1:
                    load_type = "сжатие и растяжение"
                    break
                else:
                    load_type = "no type load"

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
        # Добавляем в таблицу столбцы с названием месторождения и скважины:
        # Если слова месторождение и скважины отделены друг от друга запятой
        if len(identification) == 2:
            data["Месторождение"] = identification[0]
            data["Скважина"] = identification[1]
        else:
            # Если слова месторождение и скважины не отделены друг от друга запятой
            data["Месторождение"] = "".join(identification)
            if identification[0].find("кважин") > -1:
                fin_index = identification[0].find("кважин")
                data["Скважина"] = identification[0][fin_index - 1:]
        # data['ДО'] = company  # добавляем название дочернего общества

        return data, experiment_type, load_type

    except Exception as error:
        print(f"В файле {path}")
        print("Ошибка при вызове функции:", error)
        return None


def findall(string, char):
    """
    Переписали встроенную функцию поиска определёненого символа в строке.
    Возвращает позиции символов
    """
    pos = []
    for i in range(len(string)):
        if char == string[i]:
            pos.append(i)
    return pos


def create_df(st_pathlist):
    """
    Функция принимает список путей до файлров с прочностными исследованиями.
    Возвращает слоарь датафреймов, с данными из этих файлов
    """
    try:
        big_data_tuple = {}
        big_data = {}
        extype = {}
        lotype = {}

        i = 0
        for name in st_pathlist:
            #     print(name)
            big_data_tuple[i] = get_data_st(name)
            big_data[i] = big_data_tuple[i][0]
            extype[i] = big_data_tuple[i][
                1
            ]  # читаем тип экспетимента (сжатие или растяжение)
            lotype[i] = big_data_tuple[i][
                2
            ]  # читаем тип нагрузки (одноосная, трёхосная и тд)
            if (
                lotype[i] == "no type load"
            ):  # если не нашли тип нагрухки в файле смотрим
                # на название файла и вычленяем тип нагрузки от туда
                path_to_file = st_pathlist[i]
                word_start = findall(path_to_file, "\\")[-1] + 1
                word_end = path_to_file.find(".xl")
                word = path_to_file[word_start:word_end]

                ed1 = []

                for ch in word:
                    if (ch.isdigit() or ch == "-" or ch == "_") and word.index(
                        ch
                    ) < len(word):
                        ed1.append(word.index(ch))
                if len(ed1) > 0:
                    edm = min(ed1)
                    word = word[:edm]
                lotype[i] = word.lower()

            if lotype[i] == "no type load":
                lotype[i] = ""
            if extype[i] == "no type experiment":
                extype[i] = ""

            i += 1
        return big_data, extype, lotype

    except Exception as error:
        print(f"Ошибка при вызове функции create_df(): {error}")
        return None


def load_and_cols(big_data, lotype):
    """Сшиваем многоуровневые хедеры до одноуровневых
    Добовляем стобец "Тип нагрузки" """
    for i in range(len(big_data)):
        col = big_data[i].columns[0][:]
        # Убираем строку "Продолжение" если она есть
        try:
            #         print(BigData[i][col])
            word = big_data[i][col].str.find("родолже")
            idx = list(word).index(1.0)
            idx = [idx, idx + 1]
            big_data[i] = big_data[i].drop(idx)

        except:
            try:
                w = big_data[i].shape[1] - 3
                col = big_data[i].columns[w][:]
                s = big_data[i][col].str.find("родолже")
                idx = list(s).index(1.0)
                idx = [idx, idx + 1]
                big_data[i] = big_data[i].drop(idx)
            except:
                continue

        finally:
            big_data[i]["Тип нагрузки"] = lotype[i]
            # cc=BigData[i].columns[0]
            # el=BigData[i].loc[0,cc]
            # if isinstance(el, np.int64):
            big_data[i] = big_data[i].drop(0)

            big_data[i] = big_data[i].reset_index(drop=True)
    #         print()
    return big_data


def header_clean_st(big_data, extype, lotype):
    """Функция принимает словарь датафреймов, соединяет двууровневые столбцы,
    Добавляет в столбец "предел прочности" тип исследования, заменяет пропуски
    в названии столбцов на пропуски длины, указанной в словаре"""
    for i in range(len(big_data)):
        df = big_data[i]
        head = list(df.columns)
        hh = []
        if isinstance(head[0], str):
            for item_head in head:
                if item_head.find("реде") >= 0:
                    item_head = item_head + " " + extype[i] + " " + lotype[i]
                hh.append(item_head)

        if isinstance(head[0], tuple):
            for item_head_t in head:
                if item_head_t[1].find("name") >= 0:
                    head_new = item_head_t[0]
                    if head_new.find("реде") >= 0:
                        head_new = head_new + " " + extype[i] + " " + lotype[i]
                elif item_head_t[0] == item_head_t[1] or item_head_t[1] == " ":
                    head_new = item_head_t[0]
                    if head_new.find("реде") >= 0:
                        head_new = head_new + " " + extype[i] + " " + lotype[i]
                elif item_head_t[1].find("ровл") > 0 or item_head_t[1].find("одошв") > 0:
                    head_new = item_head_t[0] + item_head_t[1]
                else:
                    head_new = item_head_t[0] + item_head_t[1]

                if head_new.find("  ") >= 0:
                    head_new = head_new.replace("  ", " ")
                if head_new.find("            ") >= 0:
                    head_new = head_new.replace("            ", "      ")
                hh.append(head_new)

        df.columns = hh

    return big_data


def comp_stretching(big_data, lotype):
    """ обратаываем файлы, где есть "сжатие и растяжение". Создаём группы со сходными значениями параметров,
    делаем замену в столбцах 'σсж' и 'σр', если их значения отсутствуют, на значения из этой же группы """
    for i in range(0, len(big_data)):
        if lotype[i] == "сжатие и растяжение":
            #                 print('eeee')
            big_data[i]["Группа"] = ""
            #         print(i)
            # testName = "Проницаемость, мД"

            colum_bd = big_data[i].columns
            for name in colum_bd:
                if name.find("σсж"):
                    comp = name
                elif name.find("σр"):
                    stret = name

            orig1 = big_data[i][comp].copy()
            orig2 = big_data[i][stret].copy()
            leng = big_data[i].shape[0]
            for nomber in range(leng):
                big_data[i].at[nomber, "Группа"] = nomber // 2

            for group in range(leng // 2):
                idx = big_data[i].index[big_data[i]["Группа"] == group].tolist()
                up_cell = idx[0]
                down_cell = idx[1]

                for colname in big_data[i].columns:
                    #                 print(colname)
                    try:
                        if np.isnan(big_data[i][colname][up_cell]) is True:
                            big_data[i].at[up_cell, colname] = big_data[i][colname][
                                down_cell
                            ].copy()
                    except:
                        continue
                    try:
                        if np.isnan(big_data[i][colname][down_cell]) is True:
                            big_data[i].at[down_cell, colname] = big_data[i][colname][
                                up_cell
                            ].copy()
                    except:
                        continue
            big_data[i]["σсж, МПа___orig"] = orig1
            big_data[i]["σр, МПа___orig"] = orig2
        else:
            big_data[i]["Группа"] = 0

        head = list(big_data[i].columns)
        new_col = []
        for h in head:
            if h.find("кважин") >= 0:
                for nomber in range(len(big_data[i])):
                    idm = big_data[i].at[nomber, h]
                    gr = big_data[i].at[nomber, "Группа"]
                    new_col.append(str(gr) + "__" + idm)
                #                     print(idm+str(gr))
                big_data[i]["Группа"] = new_col

    return big_data


def dict_head(df, Dic):
    """ Функция получает на вход датафрейм и словарь.
     На выходе список новых хедеров согласно словарю"""

    new_dic_header = []

    for name in Dic["header"]:
        copy_name = copy.copy(name)
        if isinstance(copy_name, str):
            if copy_name.find("#") >= -1:
                copy_name = copy_name.replace("#", "")
            if copy_name[-1] == "_":
                copy_name = copy_name.replace("_", "")
            if copy_name.find("_"):
                copy_name = copy_name.replace("_", " ")
            copy_name = copy_name.strip()
        new_dic_header.append(copy_name)

    Dic["header"] = new_dic_header

    head = df.columns

    heders_in_oneline = []
    for h in head:
        w = h.replace("\n", " ")
        heders_in_oneline.append(w)

    df.columns = heders_in_oneline

    new_header = []
    for col in df.columns:
        copy_name = copy.copy(col)
        copy_name = copy_name.strip()
        for DicName in Dic["header"]:
            if copy_name == DicName:
                ind = Dic.index[Dic["header"] == DicName].tolist()
                copy_name = Dic["Словарь"][ind[0]]
        new_header.append(copy_name)
    return new_header


def use_dict(big_data, dic):
    """ Обрабатываем названия с помощью словаря и удаляем NaN"""
    for i in range(len(big_data)):
        hh = dict_head(big_data[i], dic)
        big_data[i].columns = hh
        first = big_data[i].columns[1]

        idx = big_data[i][first][big_data[i][first].isna() == True].index
        if len(idx) > 0:
            ind = idx[0]
            big_data[i] = big_data[i][0:ind]
    return big_data


def strength_combine(df_list):
    """Функция принимает список датафреймов по прочностным исследованиям (df_list),
    содержащий все данные, относящиеся к одной скважине.
    Вычисляет глубину по бурению и объединяет их в общий датафрейм.
    Если не удалось обработать таблицы, возвращает None и справочно выводит
    типы данных по столбцам и значения первой строки в каждом датафрейме списка."""
    try:
        # Если в списке всего один датафрейм, просто возвращаем его,
        # добавив столбец с глубиной по бурению:
        if len(df_list) == 1:
            strength_df = df_list[0]
            # Преобразуем столбцы типа 'object' в числовые:
            strength_df.loc[:, "Интервал отбора кровля, м"] = pd.to_numeric(
                strength_df.loc[:, "Интервал отбора кровля, м"], errors="coerce"
            )
            strength_df.loc[:, "Место взятия от верха, м"] = pd.to_numeric(
                strength_df.loc[:, "Место взятия от верха, м"], errors="coerce"
            )
            # Добавляем столбец с глубиной по бурению:
            strength_df["Глубина отбора по бурению, м"] = (
                strength_df["Интервал отбора кровля, м"]
                + strength_df["Место взятия от верха, м"]
            )
            # Если в таблице есть столбцы с одинаковыми названиями, удаляем дубликаты:
            strength_df = strength_df.loc[:, ~strength_df.columns.duplicated()]
            # Добавляем префикс, указывающий на тип файлов:
            strength_df = strength_df.add_prefix("S_")
            return strength_df

        elif len(df_list) == 0:  # Если в списке нет датафреймов
            return []

        else:  # Если в списке несколько датафреймов:
            # Проверяем, что в датафреймах нет дублирующихся столбцов в одной таблице:
            for i in range(len(df_list)):
                data = df_list[i]
                df_list[i] = data.loc[:, ~data.columns.duplicated()]
            # Соединяем все датафреймы в один:
            strength_df = pd.concat(df_list, sort=False)

            # Преобразуем столбцы типа 'object' в числовые:
            strength_df.loc[:, "Интервал отбора кровля, м"] = pd.to_numeric(
                strength_df.loc[:, "Интервал отбора кровля, м"], errors="coerce"
            )
            strength_df.loc[:, "Место взятия от верха, м"] = pd.to_numeric(
                strength_df.loc[:, "Место взятия от верха, м"], errors="coerce"
            )

            # Добавляем столбец с глубиной по бурению:
            strength_df["Глубина отбора по бурению, м"] = (
                strength_df["Интервал отбора кровля, м"]
                + strength_df["Место взятия от верха, м"]
            )
            # Выбираем заголовки столбцов с числовым типом данных:
            num_cols = strength_df.select_dtypes(include="number").columns
            # Выбираем заголовки столбцов с нечисловыми данными:
            other_cols = strength_df.select_dtypes(exclude="number").columns
            # Составляем словарь заголовков для аггрегирования данных:
            agg_dict = dict()
            for col in num_cols:  # для числовых столбцов берем среднее значение
                agg_dict[col] = "mean"
            for (
                col
            ) in other_cols:  # для прочих столбцов берем последнее текстовое значение
                agg_dict[col] = "max"
                strength_df[col] = strength_df[col].astype(str)
            # Группируем данные по глубине и аггрегируем по словарю:
            strength_df = pd.DataFrame(
                strength_df.groupby("Глубина отбора по бурению, м")
                .agg(agg_dict)
                .reset_index(drop=True)
            )
            # Добавляем префикс, указывающий на тип файлов:
            strength_df = strength_df.add_prefix("S_")
            return strength_df

    except Exception as e:
        print(f"Ошибка при вызове функции strength_combine(): {e}")
        for table in df_list:
            print("Типы данных:")
            print(table.dtypes)
        return []


def main_str(path, pathdict):
    """Функция принимает путь к папке скважины и запускает
    вспомогательные функции для поиска и обработки данных по прочности."""
    st_pathlist = file_search_st(path)  # ищем файлы по ключевым словам
    if st_pathlist is None:
        return []

    dataframe = create_df(st_pathlist)
    if dataframe is None:
        return []
    big_data, extype, lotype = dataframe
    big_data1 = load_and_cols(big_data, lotype)
    big_data2 = header_clean_st(big_data1, extype, lotype)
    big_data3 = comp_stretching(big_data2, lotype)

    xls = pd.ExcelFile(pathdict)
    sheets = xls.sheet_names
    OurDic = pd.read_excel(pathdict, sheet_name=sheets[5])
    Dic = OurDic[["header", "Словарь"]].copy()

    big_data4 = use_dict(big_data3, Dic)
    # BigData4[1]
    # Собираем датафреймы по прочности из 1 скважины в список:
    list_of_strength = []
    for key in big_data4:
        list_of_strength.append(big_data4[key])

    strength_combined = strength_combine(list_of_strength)
    strength_combined["S_path"] = ("; ").join(st_pathlist)

    return strength_combined


def str_full():
    """Функция принимает путь к папке месторождения.
    Возвращает объединенный датафрейм файлов прочности."""
    DatasetStr = pd.DataFrame()
    for well in SUBLEVEL_2:
        try:
            str_df = main_str(well, PATH_DICT)
            if type(str_df) != list:
                str_df["S_wellName"] = well.split("\\")[-1]
                DatasetStr = pd.concat([DatasetStr, str_df])
                print(well + " данные по прочности добавлены")
            else:
                print(well + " данные по прочности не найдены")
        except Exception as error:
            print(f"В папке {well} прочность не взялась")

    DatasetStr.dropna(axis="columns", how="all", inplace=True)
    DatasetStr["S_Глубина отбора по бурению, м"] = DatasetStr["S_Глубина отбора по бурению, м"].round(1)
    return DatasetStr

# Создаёт объединенный датафрейм прочности
Strength = str_full()
Strength.to_excel("StrengthDataset.xlsx", index = False)
print("Датасет по прочности создан")