"""

Autor: Adrian Goik

Problem: Wykorzystanie sieci neuronowych do klasyfikacji danych w różnych problemach.

Instrukcja użycia:
1. Upewnij się, że wszystkie wymagane biblioteki są zainstalowane.
2.Pobierz wymagane zbiory danych z linków w pliku readme
3. Dostosuj ścieżki do zbiorów danych (np. `base_dir` itd.).
4. Uruchom kod, aby przeprowadzić klasyfikację na różnych zbiorach danych.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib
matplotlib.use('Agg')  # Wyłączenie interaktywnego backendu Matplotlib - problem fixing

# Funkcje do rysowania macieży pomyłek
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
       Wyświetla macierz pomyłek w terminalu jako DataFrame.

       Parametry:
       y_true (array-like): Prawdziwe etykiety klas.
       y_pred (array-like): Przewidziane etykiety klas.
       class_names (list): Lista nazw klas odpowiadających etykietom.
       title (str): Tytuł wyświetlanej macierzy pomyłek.

       Zwraca:
       None: Funkcja wyświetla macierz pomyłek w terminalu.
       """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"\n{title}")
    print(cm_df)
# Funkcja: Wyświetlanie raportu klasyfikacji
def print_classification_report(y_true, y_pred, class_names):
    """
       Wyświetla raport klasyfikacji w terminalu.

       Parameters:
       y_true (array): Prawdziwe etykiety.
       y_pred (array): Przewidywane etykiety.
       class_names (list): Nazwy klas.

       Returns:
       None
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

def print_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"\n{title}:")
    print(cm_df)

# Task 1: 1. Wykorzystać jeden z zbiorów danych z poprzednich ćwiczeń i naucz sieć neuronową.
# Porównaj skuteczność obu podejść. Dodaj logi/print screen do repozytorium.
# Załadowanie datasetu ionosphere
data_path = "ionosphere/ionosphere.data"
column_names = [f'feature_{i}' for i in range(34)] + ['class']
ionosphere_data = pd.read_csv(data_path, header=None, names=column_names)

# Przygotowanie danych
X = ionosphere_data.iloc[:, :-1]
y = ionosphere_data['class']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Zmniejsz rozmiar danych do 10% oryginału
subset_size = int(X_train.shape[0] * 0.1)
X_train_small, y_train_small = X_train[:subset_size], y_train[:subset_size]
X_test_small, y_test_small = X_test[:subset_size], y_test[:subset_size]

# Neural Network
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_nn = model_nn.fit(X_train_small, y_train_small, epochs=20, validation_split=0.2, batch_size=16)

# Test NN model
y_pred_nn = (model_nn.predict(X_test_small) > 0.5).astype("int32").flatten()
test_accuracy_nn = np.mean(y_pred_nn == y_test_small)
print(f"Neural Network Test Accuracy: {test_accuracy_nn:.2f}")
plot_confusion_matrix(y_test_small, y_pred_nn, class_names=encoder.classes_, title="Confusion Matrix - Neural Network")

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_small, y_train_small)

y_pred_lr = log_reg.predict(X_test_small)
test_accuracy_lr = np.mean(y_pred_lr == y_test_small)
print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.2f}")
plot_confusion_matrix(y_test_small, y_pred_lr, class_names=encoder.classes_, title="Confusion Matrix - Logistic Regression")

# Task 2: Train a neural network
# Ścieżka do bazy danych
base_dir = "Animals"

# Przygotowanie danych
def prepare_data(base_dir, target_size=(64, 64), batch_size=32):
    """
      Przygotowanie danych do klasyfikacji obrazów.

      Parameters:
      base_dir (str): Ścieżka do katalogu danych.
      target_size (tuple): Rozmiar obrazów.
      batch_size (int): Liczba próbek na batch.

      Returns:
      tuple: Obiekty generatorów danych (treningowy, walidacyjny).
      """
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2  # Podział na trening i walidację
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

# Przygotuj dane
train_generator, val_generator = prepare_data(base_dir)

# Pobierz nazwy klas
class_names = list(train_generator.class_indices.keys())

# Budowa sieci neuronowej
model_animals = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')  # Liczba wyjść = liczba klas
])

model_animals.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history_animals = model_animals.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Predykcja na zbiorze walidacyjnym
y_pred = model_animals.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Wyświetlenie wyników
print(f"\nTest Accuracy: {np.mean(y_pred_classes == y_true):.2f}")

# Drukowanie macierzy pomyłek w terminalu
def print_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Wyświetla macierz pomyłek w terminalu.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"\n{title}:")
    print(cm_df)

print_confusion_matrix(y_true, y_pred_classes, class_names, title="Confusion Matrix - Animals Dataset")

# Opcjonalne zapisanie wykresu macierzy pomyłek
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix - Animals Dataset")
plt.savefig("Confusion_Matrix_Animals.png")

# Task 3: Train a neural network on Fashion-MNIST
(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = fashion_mnist.load_data()

X_train_fashion = X_train_fashion / 255.0
X_test_fashion = X_test_fashion / 255.0

# Zmniejsz rozmiar danych do 10% oryginału
subset_size = int(X_train_fashion.shape[0] * 0.1)
X_train_small_fashion, y_train_small_fashion = X_train_fashion[:subset_size], y_train_fashion[:subset_size]
X_test_small_fashion, y_test_small_fashion = X_test_fashion[:subset_size], y_test_fashion[:subset_size]

model_fashion = Sequential([
    Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model_fashion.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train_flat_fashion = X_train_small_fashion.reshape(X_train_small_fashion.shape[0], -1)
X_test_flat_fashion = X_test_small_fashion.reshape(X_test_small_fashion.shape[0], -1)

history_fashion = model_fashion.fit(X_train_flat_fashion, y_train_small_fashion, epochs=10, validation_split=0.2, batch_size=32)

y_pred_fashion = np.argmax(model_fashion.predict(X_test_flat_fashion), axis=1)
test_accuracy_fashion = np.mean(y_pred_fashion == y_test_small_fashion)
print(f"Fashion-MNIST Neural Network Test Accuracy: {test_accuracy_fashion:.2f}")

fashion_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
plot_confusion_matrix(y_test_small_fashion, y_pred_fashion, class_names=fashion_classes, title="Confusion Matrix - Fashion-MNIST")

# Task 4: Rozpoznawanie czy na zdjęciu jest mikołaj
# Ścieżka do zbioru danych
base_dir = "isThatSanta"  # Zamień na właściwą ścieżkę

# Przygotowanie danych
def prepare_data(base_dir, target_size=(128, 128), batch_size=32):
    """
    Przygotowanie danych do klasyfikacji.
    """
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2  # Podział na trening i walidację
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

train_generator, val_generator = prepare_data(base_dir)

# Budowa modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Warstwa wyjściowa dla klasyfikacji binarnej
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    batch_size=32
)

# Predykcja na danych walidacyjnych
y_pred = (model.predict(val_generator) > 0.5).astype("int32").flatten()
y_true = val_generator.classes

# Ewaluacja
class_names = ['Not Santa', 'Santa']
print_classification_report(y_true, y_pred, class_names)
print_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix - Santa Dataset")




