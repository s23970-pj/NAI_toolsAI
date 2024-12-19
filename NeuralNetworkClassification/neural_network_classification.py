import matplotlib.pyplot as plt
import mnist_reader
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def prepare_data(data: DataFrame, label_column_name: str, has_header: bool):
    """
    Przygotowuje dane do nauczenia sieci neuronowej poprzez nadanie nagłówków jeżeli ich nie ma oraz przeniesienie
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


def neural_network(data: DataFrame, input_neurons: int):
    """
    Tworzy sieć neuronową i wyświetla informacje o niej.
    :param data: Dane wejściowe
    :param input_neurons: Liczba neuronów w warstwie wejściowej. Liczba neuronów w kolejnej warstwie to połowa tej wartości.
    """
    # Podział na cechy oraz etykiety
    X = data.iloc[:, :-1]  # Cechy (wszystkie kolumny oprócz ostatniej)
    y = data.iloc[:, -1]  # Etykiety (ostatnia kolumna)

    # Zamiana etykiet w formie tekstowej na wartości liczbowe
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Podział na dane treningowe oraz testowe (80% do 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Standaryzacja danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tworzenie modelu sekwencyjnie - kolejne warstwy dodawane jedna po drugiej
    model = Sequential([
        Dense(input_neurons, activation='relu', input_shape=(X_train.shape[1],)), # Warstwa wejściowa z funkcją aktywacji ReLU i rozmiarem danych wejściowych odpowiadającym liczbie kolumn w danych treningowych
        Dense(int(input_neurons / 2), activation='relu'),  # Warstwa ukryta
        Dense(len(np.unique(y_encoded)), activation='softmax') # Warstwa wyjściowa z trzema neuronami, co odpowiada ilości klas
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trenowanie modelu
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

    # Testowanie
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # Wizualizacja wyników
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Ładowanie danych
stars_dataset_path = "../Classification/data/star_classification/star_classification.csv"

stars_data = pd.read_csv(stars_dataset_path)
stars_data = stars_data[['u', 'g', 'r', 'i', 'z', 'redshift', 'class']]
star_stats = stars_data.describe()

# CIFAR-10
(cf_x_train, cf_y_train), (cf_x_test, cf_y_test) = tf.keras.datasets.cifar10.load_data()

cf_x_train_flat = cf_x_train.reshape(cf_x_train.shape[0], -1)  # "Spłaszczenie" obrazów
cifar10_data = pd.DataFrame(cf_x_train_flat)

cifar10_data['label'] = cf_y_train

# fashion-mnist
# fm_X_train, fm_y_train = mnist_reader.load_mnist('data/fashion', kind='train')
# fm_X_test, fm_y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Przygotowanie danych
prepared_stars_df = prepare_data(stars_data, "class", True)
prepared_cifar10_df = prepare_data(cifar10_data, "label", True)

print("Stars:")
print(prepared_stars_df.head())
print()

neural_network(prepared_stars_df, 64)

print("CIFAR-10:")
print(prepared_cifar10_df.head())
print()

neural_network(prepared_cifar10_df, 128)
