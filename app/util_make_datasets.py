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


#########################################################
# Подготовка данных
#########################################################
def add_user_features(df_ratings):
    """Функция рассчитывает расширенные признаки пользователей."""
    # Средняя оценка пользователя
    user_avg_rating = df_ratings.groupby('user_id')['rating'].mean().rename('avg_user_rating')
    # Количество оценок пользователя
    user_num_ratings = df_ratings.groupby('user_id')['rating'].count().rename('num_user_ratings')
    # Активность пользователя
    total_unique_books = len(df_ratings['book_id'].unique())
    user_activity = user_num_ratings / total_unique_books * 100
    user_activity.name = 'user_activity'
    
    df_users = pd.concat([user_avg_rating, user_num_ratings, user_activity], axis=1)
    df_users = df_users.reset_index()
    return df_users

def add_book_features(df_ratings, df_books, df_book_tags):
    """Функция рассчитывает расширенные признаки книг."""
    # Популярность книги
    book_popularity = df_ratings.groupby('book_id')['user_id'].nunique().rename('popularity').to_frame()
    # Разнообразие оценок (стандартное отклонение)
    book_rating_std = df_ratings.groupby('book_id')['rating'].std().rename('rating_std').to_frame()
    # Тематическая категория
    # Отбираем самый популярный тэг
    top_tag_per_book = df_book_tags.groupby('goodreads_book_id')['count'].idxmax()
    df_top_tags = df_book_tags.iloc[top_tag_per_book].copy()
    df_top_tags.rename(columns={'tag_id': 'top_tag', 'goodreads_book_id': 'book_id'}, inplace=True)
    df_top_tags.drop(columns=['count'], inplace=True)
    
    # Добавляем признаки
    df_books_with_tags = df_books.merge(df_top_tags, on='book_id', how='left')
    df_books_extended = df_books_with_tags.merge(book_popularity, on='book_id', how='left')
    df_books_extended = df_books_extended.merge(book_rating_std, on='book_id', how='left')

    # Оставляем только существенные признаки
    df_books_extended = df_books_extended[['id', 'book_id', 'best_book_id', 'authors', 
                                           'original_title', 'title', 'language_code', 'average_rating', 
                                           'ratings_count', 'work_ratings_count', 'work_text_reviews_count', 
                                           'top_tag', 'popularity', 'rating_std']]
    
    return df_books_extended

def make_datasets(data_path):
    """Функция для создания и предобработки датасетов."""
    # global df_interaction_matrix
    
    # Загрузка данных
    df_ratings = pd.read_csv(data_path + '/ratings.csv')
    df_books = pd.read_csv(data_path + '/books.csv')
    df_tags = pd.read_csv(data_path + '/tags.csv')
    df_book_tags = pd.read_csv(data_path + '/book_tags.csv')

    # Предобработка с учетом предыдущей работы
    df_ratings = df_ratings.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()

    # Создание расширенных признаков
    df_users = add_user_features(df_ratings)
    df_books = add_book_features(df_ratings, df_books, df_book_tags)

    return df_ratings, df_books, df_tags, df_book_tags, df_users

#########################################################
# Запуск
#########################################################
if __name__ == '__main__':
    load_dotenv()
    
    data_path = os.path.abspath(os.getenv('data_path'))

    # Загружаем даныне
    print('Загружаем данные...')
    df_ratings, df_books, df_tags, df_book_tags, df_users = make_datasets(data_path)
    print(df_ratings.info())
    print(df_books.info())
    print(df_tags.info())
    print(df_book_tags.info())
    print(df_users.info())