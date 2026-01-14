import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from dotenv import load_dotenv
from util_make_datasets import make_datasets
from util_load_best_params_svd import load_best_params_svd
from util_load_interaction_matrix import create_similarity_matrix, load_interaction_matrix


#########################################################
# Модели
#########################################################
def get_popularity_recommendation_ids(df_ratings, N=5):
    """Модель популярности (топ-N популярных книг).
Функция показывает топ-N самых популярных книг по количеству оценок."""
    # Получаем топ-N самых популярных книг
    popular_books = df_ratings['book_id'].value_counts().index[:N]

    popular_book_ids = list(popular_books)

    # Возвращаем id топ-N книг
    return popular_book_ids

def get_similar_books_ids(df_book_tags, df_tags, df_books, df_ratings, book_id=None, N=5):
    """Контентная модель (похожие книги по тегам и названиям).
Функция для поиска похожих книг по косинусной мере близости между TF-IDF векторами."""
    # Проверка
    if book_id is None \
        or book_id not in set(df_book_tags['goodreads_book_id']) \
        or book_id not in set(df_books['book_id']):
        popular_book_ids = get_popularity_recommendation_ids(df_ratings, N)
        return popular_book_ids

    global books_profile, tfidf_matrix
    try:
        books_profile, tfidf_matrix
    except NameError:
        books_profile, tfidf_matrix = create_similarity_matrix(df_book_tags, df_tags, df_books, N=5)
    finally:
        # Находим индекс книги        
        idx = books_profile[books_profile['book_id'] == book_id].index[0]
        
        # Вычисляем косинусную близость
        similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Получаем индексы N самых похожих книг
        similar_indices = similarity_scores.argsort()[-N-1:-1][::-1]
        similar_book_ids = books_profile.iloc[similar_indices]['book_id'].values.tolist()
        
        # Возвращаем id N самых похожих книг
        return similar_book_ids

def get_predict_rating_value(user_id, book_id, df_ratings, K=25):
    """Коллаборативная фильтрация (Item-Based).
Функция предсказывает оценку пользователя для заданной книги."""
    # Проверка на существование
    if user_id not in set(df_ratings['user_id']) \
        or book_id not in set(df_ratings['book_id']) \
        or book_id is None \
        or book_id == 'None':
        return 0

    # Проверка уже имеющейся оценки
    if book_id in set(df_ratings[df_ratings['user_id'] == user_id]['book_id']):
        predicted_rating = df_ratings[(df_ratings['book_id'] == book_id) & (df_ratings['user_id'] == user_id)]['rating']
        predicted_rating = float(predicted_rating.iloc[0])
        return predicted_rating

    # Отберем оцененные пользователем книги
    user_ratings = df_ratings[df_ratings['user_id'] == user_id]
    rated_books = list(user_ratings['book_id'])

    # Построим матрицу взаимодействий user_x_book, заполнив пропущенные значения нулями
    # Отфильтровываем лишние записи для эффективного использования памяти
    df_filtered = df_ratings[(df_ratings['book_id'].isin([book_id])) | (df_ratings['book_id'].isin(rated_books))]
    user_book_matrix = df_filtered.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

    # Мы помним, что большее количество пользователей ставят малое количество оценок (разреженная матрица)
    # В таком случае имеет смысл преобразовать матрицу в формат CSR, для более эффективного использования памяти и ускорения вычислений
    sparse_matrix = csr_matrix(user_book_matrix.values)

    # Рассчитаем матрицу попарных схожестей между книгами по косинусной близости по векторам оценок
    item_similarity = cosine_similarity(sparse_matrix.T)

    # Найдем наиболее похожие книги для заданной 
    target_book_col = user_book_matrix.columns.get_loc(book_id)
    similarities = item_similarity[target_book_col]
    most_similar_books = [(col, sim) for col, sim in zip(user_book_matrix.columns, similarities) if sim > 0]
    most_similar_books.sort(key=lambda x: x[1], reverse=True)
    top_k_books = most_similar_books[:K]

    # Вычисление предсказания оценки: сумма произведения оценок пользователя на схожесть между книгами делится на сумму схожестей между книгами
    numerator = 0
    for book, sim in top_k_books:
        ratings_for_book = user_ratings[user_ratings['book_id'] == book]['rating']
        if not ratings_for_book.empty:
            numerator += ratings_for_book.sum() * sim

    denominator = sum(sim for book, sim in top_k_books)

    predicted_rating = 0
    if denominator > 0:
        predicted_rating = numerator / denominator
    else:
        predicted_rating = 0

    return predicted_rating

def get_recommendations_svd(user_id, df_ratings, N=5):
    """Матричная факторизация (SVD).
Функция возвращает топ-N книг с наибольшим предсказанным рейтингом для заданного пользователя"""
    # Проверка на существование
    if user_id not in set(df_ratings['user_id']):
        popular_book_ids = get_popularity_recommendation_ids(df_ratings, N)
        return popular_book_ids

    # Проверка, рассчитывались ли гиперпараметры
    global best_params
    try:
        n_factors, n_epochs, lr_all, reg_all = best_params.values()
    except NameError:
        best_params = load_best_params_svd(params_path, df_ratings, 'increment')
        n_factors, n_epochs, lr_all, reg_all = best_params.values()
    # Обучаем модель с лучшими гиперпараметрами
    finally:
        # Загрузим данные в формат, подходящий для scikit-surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_ratings[['user_id', 'book_id', 'rating']], reader).build_full_trainset()

        # Обучим модель SVD
        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        model.fit(data)

        # Получим множество книг, которые пользователь еще не оценил
        user_items = set(data.to_raw_iid(i) for i in data.all_items())
        user_rated_items = set((data.to_raw_iid(iid) for (uid, iid, _) in data.all_ratings() if data.to_raw_uid(uid) == user_id))
        items_to_predict = user_items - user_rated_items

        # Выполним предсказание для всех книг и отсортируем их в порядке убывания оценки
        predictions = [model.predict(user_id, item) for item in items_to_predict]
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        top_books_svd = [pred.iid for pred in predictions[:N]]
        
        return top_books_svd

def get_recomendation_interaction_hist(df_interaction_matrix, df_ratings, N=5, threshold=None):
    """Признаки взаимодействий: схожесть с историей пользователя.
Функция возвращает топ-N книг наиболее схожих по истории взаимодействия пользователя"""
    # Фильтруем прочитанные
    user_id = df_interaction_matrix.index.values[0]
    read_books = set(map(str, df_ratings[df_ratings['user_id'] == user_id]['book_id']))
    unread_books = df_interaction_matrix.columns.difference(read_books)

    # Забираем либо топ, либо выше порога
    if threshold is None:
        top_books_interaction_hist = df_interaction_matrix.iloc[0][unread_books].sort_values(ascending=False).head(N).index.to_list()
    else:
        top_books_interaction_hist = unread_books[df_interaction_matrix.iloc[0][unread_books] >= threshold].tolist()

    top_books_interaction_hist = [int(i) for i in top_books_interaction_hist]

    return top_books_interaction_hist

#########################################################
# Управляющая функция
#########################################################
def main_models(data_path, book_id, user_id):
    # Находим топ популярных книг
    print('    Находим топ популярных книг...')
    popular_book_ids = get_popularity_recommendation_ids(df_ratings)

    # Находим топ книг похожих на заданную
    print('    Находим топ книг похожих на заданную...')
    similar_book_ids = get_similar_books_ids(df_book_tags, df_tags, df_books, df_ratings, book_id)

    # Находим предположительную оценку книги для пользователя
    print('    Находим предположительную оценку книги для пользователя...')
    predict_rating = get_predict_rating_value(user_id, book_id, df_ratings)

    # Рассчитываем наиболее подходящие книги (SVD)
    print('    Рассчитываем наиболее подходящие книги (SVD)...')
    top_books_svd = get_recommendations_svd(user_id, df_ratings)

    # Рассчитываем наиболее подходящие книги (История взаимодействий)
    print('    Рассчитываем наиболее подходящие книги (История взаимодействий)...')
    top_books_interaction_hist = get_recomendation_interaction_hist(df_interaction_matrix, df_ratings)

    print('...')
    return popular_book_ids, similar_book_ids, predict_rating, top_books_svd, top_books_interaction_hist
#########################################################
# Запуск
#########################################################
# Определяем запуск из-под скрипта:
if __name__ == '__main__':
    load_dotenv()

    data_path = os.path.abspath(os.getenv('data_path'))
    params_path = os.path.abspath(os.getenv('params_path'))
    matrix_folder = os.path.abspath(os.getenv('matrix_path'))
    matrix_name = os.getenv('matrix_name')

    # Загружаем даныне
    print('Загружаем даныне...')
    df_ratings, df_books, df_tags, df_book_tags, df_users = make_datasets(data_path)

    # Обучаем модель SVD
    print('Обучаем модель SVD...')
    best_params = load_best_params_svd(params_path, df_ratings, mode='increment')

    while True:
        user_id = int(input('Введите ID пользователя: '))

        matrix_path = f'{matrix_folder}/{user_id}{matrix_name}'

        print('Оцениваем историю взаимодействий...')
        df_interaction_matrix = load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment')

        try:
            book_id = input('Введите ID книги (при отсутствии - None):')
            book_id = int(book_id)
        except ValueError:
            book_id = None
        
        print('Рассчитываем рекомендации...')
        popular_book_ids, similar_book_ids, predict_rating, top_books_svd, top_books_interaction_hist = main_models(data_path, book_id, user_id)
    
        print('Топ популярных книг:')
        print(popular_book_ids)
        print('Книги, похожие по текстовому профилю:')
        print(similar_book_ids)
        print('Вероятная оценка пользователя для книги:')
        print(predict_rating)
        print('Наиболее подходящие книги для пользователя (SVD):')
        print(top_books_svd)
        print('Наиболее подходящие книги для пользователя (История взаимодействий):')
        print(top_books_interaction_hist)
