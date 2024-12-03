import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree


def prepare_data(data: DataFrame, label_column_name: str, has_header: bool):
    """
    Przygotowuje dane do stworzenia drzewa decyzyjnego poprzez nadanie nagłówków jeżeli ich nie ma oraz przeniesienie
    kolumny z etykietami na koniec DataFrame-u.
    :param data: Dane wejściowe
    :param label_column_name: Nazwa kolumny z etykietami (jeżeli obecna) - jeżeli nieobecna to zostanie utworzona
    :param has_header: Czy DataFrame zawiera nagłówek
    :return: Przystosowany DataFrame
    """
    if has_header:  # Jeżeli DataFrame zawiera nagłówek to przenieś kolumnę etykiet o danej nazwie na jego koniec
        if label_column_name in data.columns:
            label_column = data.pop(label_column_name)
            data[label_column_name] = label_column
    else:  # W innym przypadku traktuj ostatnią kolumnę jako etykiety i nazwij ją
        X = data.iloc[:, :-1]

        columns = [f"feature_{i}" for i in range(X.shape[1])] + [
            label_column_name]  # Nazwanie kolumn i przeniesienie etykiet na koniec
        data.columns = columns

    return data


def decision_tree(data: DataFrame, export_file_suffix: str):
    """
    Tworzy drzewo decyzyjne, wyświetla informacje o nim oraz eksportuje jego graf.
    :param data: Dane wejściowe
    :param export_file_suffix: Sufiks dodawany do nazwy eksportowanego pliku
    """
    # Podział na cechy oraz etykiety
    X = data.iloc[:, :-1]  # Cechy (wszystkie kolumny oprócz ostatniej)
    y = data.iloc[:, -1]  # Etykiety (ostatnia kolumna)

    # Podział na dane treningowe oraz testowe (80% do 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja drzewa decyzyjnego
    tree = DecisionTreeClassifier()

    # Dopasowanie modelu do danych treningowych
    tree.fit(X_train, y_train)

    # Prognoza na podstawie danych testowych
    y_predict = tree.predict(X_test)

    # Ocena trafności
    accuracy_tree = accuracy_score(y_test, y_predict)
    print(f"Accuracy of decision tree: {accuracy_tree}")

    # Wizualizacja drzewa
    plt.figure(figsize=(20, 12))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=y.unique().astype(str),
        filled=True
    )
    plt.show()

    # Zapis grafu do pliku
    dot_data = export_graphviz(
        tree,
        feature_names=X.columns,
        class_names=y.unique().astype(str),
        filled=True,
        rounded=True,
        special_characters=True,
        out_file=None
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"./graphs/decision_tree_{export_file_suffix}", format="png")
    print(f"File decision_tree_{export_file_suffix} exported")


# Ładowanie danych
ionosphere_dataset_path = "./data/ionosphere/ionosphere.data"
stars_dataset_path = "./data/star_classification/star_classification.csv"

ionosphere_data = pd.read_csv(ionosphere_dataset_path, header=None)
stars_data = pd.read_csv(stars_dataset_path)

# Przygotowanie danych
prepared_ionosphere_df = prepare_data(ionosphere_data, "label", False)
prepared_stars_df = prepare_data(stars_data, "class", True)

print("Ionosphere:")
print(prepared_ionosphere_df.head())
print("Stars:")
print(prepared_stars_df.head())

# Tworzenie drzew decyzyjnych dla każdego zestawu danych
decision_tree(prepared_ionosphere_df, "ionosphere")
decision_tree(prepared_stars_df, "stars")
