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
import individual_models as im


#########################################################
# Построение гибридной модели
## Объединение моделей
#########################################################
def get_hybrid_recommendation(user_id, 
                              df_ratings, df_books, df_tags, df_book_tags, df_interaction_matrix,
                              book_id=None, weights=None):
    """Функция объединяет прогнозы разных моделей и выводит список рекомендуемых книг.
    :param user_id: идентификатор пользователя
    :param book_id: идентификатор книги
    :param df_ratings: DataFrame с рейтингами
    :param df_books: DataFrame с информацией о книгах
    :param df_tags: DataFrame с тегами
    :param df_book_tags: DataFrame с привязанными тегами к книгам
    :param weights: весовые коэффициенты для моделей [популярность, похожесть, коллаборативная фильтрация]

    :return: список рекомендованных книг"""
    # Определим веса для каждой из рекомендаций
    if weights is None:
        weights = [2, 3, 4, 5]
    
    # Получаем рекомендации книг по каждой модели в соответствии с весами
    popularity_rec       = im.get_popularity_recommendation_ids(df_ratings, weights[0])
    content_rec          = im.get_similar_books_ids(df_book_tags, df_tags, df_books, df_ratings, book_id, weights[1])
    collaborative_rec    = im.get_recommendations_svd(user_id, df_ratings, weights[2])
    hist_interaction_rec = im.get_recomendation_interaction_hist(df_interaction_matrix, df_ratings, weights[3])
    
    # Отбираем уникальные книги
    recommendations = (popularity_rec + content_rec + collaborative_rec + hist_interaction_rec)
    unique_recs = list(set(recommendations))
    
    return unique_recs

def get_user_type(user_id, df_users, threshold=None):
    """Функция классификацириует тип пользователя: новый или активный.
Основывается на количестве прочтённых книг и средней оценке.
    :param user_id: идентификатор пользователя
    :param df_users: DataFrame c информацией о пользователях

    :return: строка ('new' или 'active')
    """
    # Находим характеристики пользователя
    avg_user_rating = df_users[df_users['user_id'] == user_id]['avg_user_rating'].iloc[0]
    num_user_ratings = df_users[df_users['user_id'] == user_id]['num_user_ratings'].iloc[0]
    user_activity = df_users[df_users['user_id'] == user_id]['user_activity'].iloc[0]
    
    # Определяем погоровые значения
    if threshold is None:
        threshold = [3.5,    # Средняя оценка
                     10,     # Количество оценок
                     0.1]    # Рейтинг активности (Процентное отношение оценок к количеству книг)

    th_avg_user_rating, th_num_user_ratings, th_user_activity = threshold

    if avg_user_rating >= th_avg_user_rating and \
      (num_user_ratings >= th_num_user_ratings or user_activity >= th_user_activity):
        user_type = 'active'
    else:
        user_type = 'new'

    return user_type

def get_combined_recomendation_by_user_type(user_id, df_ratings, df_books, df_tags, df_book_tags, df_users, book_id=None, weights=None):
    """Функция комбинирует персонализированные и популярные рекомендаций на основе типа пользователя.
Для новых пользователей показываются популярные книги, активным пользователям предлагаются 
гибридные рекомендации с упором на схожесть интересов.
    :param user_id: идентификатор пользователя
    :param book_id: идентификатор книги
    :param df_ratings: DataFrame с рейтингами
    :param df_books: DataFrame с информацией о книгах
    :param df_tags: DataFrame с тегами
    :param df_book_tags: DataFrame с привязанными тегами к книгам

    :return: список рекомендованных книг
    """
    # Определяем тип пользователя
    user_type = get_user_type(user_id, df_users)

    # Новым пользователям предлагаются деперсонализированные популярные книги
    # Активным пользователям - гибридные взвешенные рекомендации
    if user_type == 'new':
        combined_recs = im.get_popularity_recommendation_ids(df_ratings, 15)
    elif user_type == 'active':
        combined_recs = get_hybrid_recommendation(user_id, 
                                                  df_ratings, df_books, df_tags, df_book_tags, df_interaction_matrix,
                                                  book_id, weights)

    return combined_recs

#########################################################
## Систем генерации кандидатов
#########################################################
def get_candidate_pool(user_id, df_ratings, df_books, df_tags, df_book_tags, book_id=None, N=5):
    """Функция объединяет рекомендации от всех моделей в общий пул с учетом уже прочитанных книг.
    :param user_id: идентификатор пользователя
    :param book_id: идентификатор книги
    :param df_ratings: DataFrame с рейтингами
    :param df_books: DataFrame с информацией о книгах
    :param df_tags: DataFrame с тегами
    :param df_book_tags: DataFrame с привязанными тегами к книгам
    
    :return: список уникальных кандидатских книг
    """
    # Получаем полные рекомендации из разных моделей
    weights = [N, N, N, N]
    candidate_pool = get_combined_recomendation_by_user_type(user_id, 
                                                             df_ratings, df_books, df_tags, df_book_tags, df_users, 
                                                             book_id, weights)
        
    # Исключаем прочитанные книги
    read_books = set(df_ratings[df_ratings['user_id'] == user_id]['book_id'])
    final_candidates = list(set(candidate_pool) - read_books)
    
    return final_candidates

def set_diversity_filter(candidates, df_books, max_genre_ratio=0.6):
    """Функция проводит балансировку между разнообразием и релевантностью,
путем ограничения доминирования определённого жанра или автора.
    :param candidates: список книг-кандидатов
    :param df_books: DataFrame с информацией о книгах
    :param max_genre_ratio: максимальный процент книг одного жанра или автора
    
    :return: сбалансированный список кандидатов
    """
    filtered_candidates = []
    tag_counts = {}
    author_counts = {}
    
    for book_id in candidates:
        row = df_books[df_books['book_id'] == book_id]
        if len(row) == 0:
            filtered_candidates.append(book_id)
            continue
        row = row.iloc[0]
        tag = row['top_tag']
        author = row['authors']
        
        # Считаем частоту появления топ-тэга и автора
        tag_counts.setdefault(tag, 0)
        author_counts.setdefault(author, 0)
        tag_counts[tag] += 1
        author_counts[author] += 1
        
        # Применяем ограничение по тэгу или автору
        if tag_counts[tag] <= len(candidates)*max_genre_ratio and author_counts[author] <= len(candidates)*max_genre_ratio:
            filtered_candidates.append(book_id)
            
    return filtered_candidates

#########################################################
# Запуск
#########################################################
# Определяем запуск из-под скрипта:
if __name__ == '__main__':
    load_dotenv()

    data_path = os.path.abspath(os.getenv('data_path'))
    params_path = im.params_path = os.path.abspath(os.getenv('params_path'))
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

        # Оцениваем историю взаимодействий
        print('Оцениваем историю взаимодействий...')
        df_interaction_matrix = load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment')

        # Получаем тип пользователя
        print('Получаем тип пользователя...')
        user_type = get_user_type(user_id, df_users)
        print(user_type)

        try:
            book_id = input('Введите ID книги (при отсутствии - None):')
            book_id = int(book_id)
        except ValueError:
            book_id = None
    
        # Получаем комбинированные рекомендации в зависимости от типа пользователя
        print('Получаем комбинированные рекомендации в зависимости от типа пользователя...')
        combined_recs = get_combined_recomendation_by_user_type(user_id, df_ratings, df_books, df_tags, df_book_tags, df_users)
        print(combined_recs)
    
        # Получаем сбалансированный список кандидатов
        print('Получаем сбалансированный список кандидатов...')
        final_candidates = get_candidate_pool(user_id, df_ratings, df_books, df_tags, df_book_tags, book_id)
        filtered_candidates = set_diversity_filter(final_candidates, df_books)
        print(filtered_candidates)