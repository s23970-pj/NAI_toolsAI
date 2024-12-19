import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import cifar10, fashion_mnist



def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Rysuje macierz pomyłek.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis')
    plt.title("Macierz Pomyłek")
    plt.show()


# Wczytanie danych
stars_data = pd.read_csv("star_classification/star_classification.csv")
stars_data = stars_data[['u', 'g', 'r', 'i', 'z', 'redshift', 'class']]

# Przygotowanie danych
X = stars_data.iloc[:, :-1]
y = stars_data.iloc[:, -1]
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tworzenie modelu
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(set(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=16)

# Testowanie i wyniki
y_pred = np.argmax(model.predict(X_test), axis=1)
test_accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Macierz pomyłek
plot_confusion_matrix(y_test, y_pred, class_names=encoder.classes_)

# Wczytanie danych
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizacja danych
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding etykiet
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Tworzenie modelu
model = Sequential([
    Dense(512, activation='relu', input_shape=(32 * 32 * 3,)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
history = model.fit(X_train_flat, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Testowanie
y_pred = np.argmax(model.predict(X_test_flat), axis=1)
test_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"Test accuracy: {test_accuracy:.2f}")

# Macierz pomyłek
cifar_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
plot_confusion_matrix(np.argmax(y_test, axis=1), y_pred, class_names=cifar_classes)

# Wczytanie danych
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalizacja
X_train = X_train / 255.0
X_test = X_test / 255.0

# Tworzenie modelu
model = Sequential([
    Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenowanie
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
history = model.fit(X_train_flat, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Testowanie
y_pred = np.argmax(model.predict(X_test_flat), axis=1)
test_accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Macierz pomyłek
fashion_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
plot_confusion_matrix(y_test, y_pred, class_names=fashion_classes)
