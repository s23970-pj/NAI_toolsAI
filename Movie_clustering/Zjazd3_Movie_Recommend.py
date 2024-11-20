import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def load_and_transform_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje i przekształca dane z pliku do formatu [User, Movie, Rating].

    Args:
        file_path (str): Ścieżka do pliku z danymi.

    Returns:
        pd.DataFrame: Dane w formacie [User, Movie, Rating].
    """
    data = pd.ExcelFile(file_path)
    df = data.parse('Arkusz1')
    users = df.iloc[:, 0]
    ratings_data = df.iloc[:, 1:]

    cleaned_data = []
    for index, row in ratings_data.iterrows():
        for col_idx in range(0, len(row), 2):  # Kolumny z filmami i ocenami
            movie = row.iloc[col_idx]  # Użycie iloc dla bezpieczeństwa
            if pd.notna(movie):
                rating = row.iloc[col_idx + 1]  # Użycie iloc dla bezpieczeństwa
                if pd.notna(rating):
                    cleaned_data.append({'User': users.iloc[index], 'Movie': movie, 'Rating': float(rating)})

    return pd.DataFrame(cleaned_data)


def prepare_similarity_and_clustering(data: pd.DataFrame, n_clusters: int = 3):
    """
    Oblicza podobieństwa między użytkownikami i przeprowadza klasteryzację.

    Args:
        data (pd.DataFrame): Dane w formacie [User, Movie, Rating].
        n_clusters (int): Liczba klastrów.

    Returns:
        Tuple[pd.DataFrame, KMeans]: Macierz użytkowników oraz model KMeans.
    """
    user_movie_matrix = data.pivot_table(
        index='User', columns='Movie', values='Rating', fill_value=0
    )
    cosine_sim = cosine_similarity(user_movie_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(cosine_sim)
    return user_movie_matrix, kmeans


def recommend_movies(data: pd.DataFrame, user: str, user_movie_matrix: pd.DataFrame, kmeans: KMeans, top_n: int = 5):
    """
    Generuje rekomendacje filmów na podstawie klastrów.

    Args:
        data (pd.DataFrame): Dane wejściowe.
        user (str): Użytkownik, dla którego generujemy rekomendacje.
        user_movie_matrix (pd.DataFrame): Macierz użytkownik-film.
        kmeans (KMeans): Model klasteryzacji.
        top_n (int): Liczba rekomendacji.

    Returns:
        pd.DataFrame: Rekomendacje filmów.
    """
    cluster_label = kmeans.labels_[list(user_movie_matrix.index).index(user)]
    movies_watched = data[data['User'] == user]['Movie']

    # Wybór filmów z tego samego klastra
    cluster_users = user_movie_matrix.index[kmeans.labels_ == cluster_label]
    same_cluster_movies = data[data['User'].isin(cluster_users) & (~data['Movie'].isin(movies_watched))]

    return same_cluster_movies.groupby('Movie')['Rating'].mean().sort_values(ascending=False).head(top_n)


def anti_recommend_movies(data: pd.DataFrame, user: str, user_movie_matrix: pd.DataFrame, kmeans: KMeans,
                          top_n: int = 5):
    """
    Generuje antyrekomendacje filmów (najniżej oceniane filmy z klastra użytkownika).

    Args:
        data (pd.DataFrame): Dane wejściowe.
        user (str): Użytkownik, dla którego generujemy antyrekomendacje.
        user_movie_matrix (pd.DataFrame): Macierz użytkownik-film.
        kmeans (KMeans): Model klasteryzacji.
        top_n (int): Liczba antyrekomendacji.

    Returns:
        pd.DataFrame: Antyrekomendacje filmów.
    """
    cluster_label = kmeans.labels_[list(user_movie_matrix.index).index(user)]
    movies_watched = data[data['User'] == user]['Movie']

    # Wybór filmów z tego samego klastra
    cluster_users = user_movie_matrix.index[kmeans.labels_ == cluster_label]
    same_cluster_movies = data[data['User'].isin(cluster_users) & (~data['Movie'].isin(movies_watched))]

    return same_cluster_movies.groupby('Movie')['Rating'].mean().sort_values(ascending=True).head(top_n)


# Ścieżka do pliku
file_path = '/Users/adriangoik/Desktop/NAI_toolsAI/Movie_clustering/Python_ML_AdrianGoik_s23970-master.xlsx'

# Wczytanie i czyszczenie danych
data = load_and_transform_data(file_path)

# Przygotowanie macierzy i klastrów
user_movie_matrix, kmeans = prepare_similarity_and_clustering(data)

# Przykładowy użytkownik
example_user = user_movie_matrix.index[0]

# Generowanie rekomendacji i antyrekomendacji
recommended_movies = recommend_movies(data, example_user, user_movie_matrix, kmeans, top_n=5)
anti_recommended_movies = anti_recommend_movies(data, example_user, user_movie_matrix, kmeans, top_n=5)

# Wyświetlenie rekomendacji i antyrekomendacji
print("Rekomendacje filmów:")
print(recommended_movies)
print("\nAntyrekomendacje filmów:")
print(anti_recommended_movies)
