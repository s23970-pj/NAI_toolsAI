'''
Autor: Adrian Goik
Cel: Klasyfikacja dwóch różnych zbiorów danych za pomocą Drzewa Decyzyjnego oraz SVM.
Analizujemy wyniki klasyfikacji, wizualizujemy dane oraz oceniamy jakość klasyfikatorów na podstawie metryk takich jak
dokładność, precyzja, czułość i F1-score.
Instrukcja użycia:
Uruchom kod w środowisku Python (np. Jupyter Notebook, PyCharm).
Kod korzysta z bibliotek takich jak pandas, numpy, matplotlib, seaborn oraz sklearn.
Przed uruchomieniem upewnij się, że wszystkie wymagane biblioteki są zainstalowane lub zainstaluj je poleceniem:
~pip install [nazwa_biblioteki]
'''
import dataset
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.svm import SVC

def svm_classifier(data: DataFrame, dataset_name: str):
    """
    Tworzy klasyfikator SVM, wyświetla informacje o nim oraz ocenia jego działanie na zbiorze danych.
    :param data: Dane wejściowe
    :param dataset_name: Nazwa zbioru danych (np. "ionosphere" lub "stars")
    """
    # Podział na cechy oraz etykiety
    X = data.iloc[:, :-1]  # Cechy (wszystkie kolumny oprócz ostatniej)
    y = data.iloc[:, -1]  # Etykiety (ostatnia kolumna)

    # Podział na dane treningowe oraz testowe (80% do 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja SVM
    svm = SVC(kernel='linear', random_state=42)

    # Dopasowanie modelu do danych treningowych
    svm.fit(X_train, y_train)

    # Prognoza na podstawie danych testowych
    y_predict = svm.predict(X_test)

    # Ocena trafności
    accuracy_svm = accuracy_score(y_test, y_predict)
    print(f"Accuracy of SVM - {dataset_name}: {accuracy_svm}")

    # Wyświetlenie raportu klasyfikacji
    print(f"\nClassification Report SVM- {dataset_name}:")
    print(classification_report(y_test, y_predict))

    # Wyświetlenie macierzy pomyłek
    print(f"\nConfusion Matrix SVM - {dataset_name}:")
    print(confusion_matrix(y_test, y_predict))

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
    print(f"Accuracy of decision tree - {export_file_suffix}: {accuracy_tree}")

    # Wyświetlenie raportu klasyfikacji
    print(f"\nClassification Report DECISION TREE - {export_file_suffix}:")
    print(classification_report(y_test, y_predict))

    # Wyświetlenie macierzy pomyłek
    print(f"\nConfusion Matrix DECISION TREE- {export_file_suffix}:")
    print(confusion_matrix(y_test, y_predict))

    # Wizualizacja drzewa
    plt.figure(figsize=(20, 12))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=y.unique().astype(str),
        filled=True
    )
    plt.title(f"Decision Tree - {export_file_suffix}")
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
ionosphere_dataset_path = "data/ionosphere/ionosphere.data"
stars_dataset_path = "data/star_classification/star_classification.csv"

ionosphere_data = pd.read_csv(ionosphere_dataset_path, header=None)
stars_data = pd.read_csv(stars_dataset_path)
stars_data = stars_data[['u', 'g', 'r', 'i', 'z', 'redshift', 'class']]
star_stats=stars_data.describe()
# Przygotowanie danych
prepared_ionosphere_df = prepare_data(ionosphere_data, "label", False)
prepared_stars_df = prepare_data(stars_data, "class", True)

print("Ionosphere:")
print(prepared_ionosphere_df.head())
print("Stars:")
print(prepared_stars_df.head())

# Tworzenie drzew decyzyjnych i SVM dla każdego zestawu danych
decision_tree(prepared_ionosphere_df, "ionosphere") #dla ionosphere
svm_classifier(prepared_ionosphere_df, "ionosphere")  # SVM
decision_tree(prepared_stars_df, "stars") # dla stars
svm_classifier(prepared_stars_df, "stars")  # SVM
