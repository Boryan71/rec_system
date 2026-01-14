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
from alive_progress import alive_bar
from util_make_datasets import make_datasets


#########################################################
# Подготовка векторизированной матрицы текстовых профилей
#########################################################
def create_similarity_matrix(df_book_tags, df_tags, df_books, N=5):
    """Функция создает матрицу tf-idf профилей книг."""
    # Создаем текстовый профиль и tf_idf-матрицу на его основе
    # Найдем названия тегов по справочнику
    cont_book_tags = pd.merge(df_book_tags, df_tags, on='tag_id', how='left')
    cont_book_tags = cont_book_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: sorted(x)).reset_index()
    
    # Соединим теги и книги
    # Создадим текстовый профиль для каждой книги
    cont_books = pd.merge(df_books, cont_book_tags, left_on='book_id', right_on='goodreads_book_id', how='left')
    cont_books['profile'] = cont_books['original_title'] + ' ' + cont_books['tag_name'].astype(str)
    books_profile = cont_books[['book_id', 'profile']]
    
    # Векторизуем текстовые проифли с помощью TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(books_profile['profile'].fillna('unknown'))

    return books_profile, tfidf_matrix

#########################################################
# Подготовка матрицы схожестей
#########################################################
def create_interaction_matrix(user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment'):
    """Функция создает матрицу схожестей книг с историей взаимодействия пользователя."""
    global df_interaction_matrix
    if mode == 'increment':
        try:
            df_interaction_matrix
        except NameError:
            mode = 'full'

    if mode == 'full':
        # Создаем текстовый профиль и TF-IDF-матрицу на его основе
        books_profile, tfidf_matrix = create_similarity_matrix(df_book_tags, df_tags, df_books)
        
        # Создаем матрицу схожестей пользователя и книг 
        books = books_profile["book_id"].unique()
        interaction_matrix = np.zeros((1, len(books)))

        # Прогресс-бар в вывод
        with alive_bar(1) as bar:
            # Оценённые книги пользователя
            rated_books = df_ratings[df_ratings["user_id"] == user_id]["book_id"]
            
            # Пропускаем, если нет оценок
            if len(rated_books) > 0:
                # Получаем индексы оценённых книг
                rated_indices = books_profile[books_profile["book_id"].isin(rated_books)].index
                
                # Рассчитываем сходство, если есть подходящие книги
                if len(rated_indices) > 0:
                    rated_tfidf = tfidf_matrix[rated_indices]
                    similarity = cosine_similarity(np.asarray(rated_tfidf.mean(axis=0)).reshape(1,-1), tfidf_matrix).flatten()
                    scaled_similarity = similarity * 100
                    
                    # Записываем в матрицу
                    interaction_matrix[0, :] = scaled_similarity
            
            bar()
        
        df_interaction_matrix = pd.DataFrame(interaction_matrix, index=[user_id], columns=books)
    
    return df_interaction_matrix

def load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment'):
    """Функция проверяет, рассчитывалась ли матрица схожести книг с взоимодействиями пользователя, и возвращает её"""
    try:
        if mode == 'increment':
            df_interaction_matrix = pd.read_csv(matrix_path, index_col=0, sep=';')
        elif mode == 'full':
            df_interaction_matrix = create_interaction_matrix(user_id, df_ratings, df_books, df_book_tags, df_tags, mode)
            df_interaction_matrix.to_csv(matrix_path, index=True, sep=";")
        else:
            raise AttributeError(f'mode={mode}. Режимы работы: full - новый расчет гиперпараметров, increment - расчет гиперпараметров, если требуется')
    except FileNotFoundError:
        df_interaction_matrix = create_interaction_matrix(user_id, df_ratings, df_books, df_book_tags, df_tags, mode)
        df_interaction_matrix.to_csv(matrix_path, index=True, sep=";")
    except AttributeError:
        raise
    finally:
        return df_interaction_matrix

#########################################################
# Запуск
#########################################################
if __name__ == '__main__':
    load_dotenv()

    user_id = int(input('Введите Ваш ID: '))
    
    data_path = os.path.abspath(os.getenv('data_path'))
    matrix_folder = os.path.abspath(os.getenv('matrix_path'))
    matrix_name = os.getenv('matrix_name')
    matrix_path = f'{matrix_folder}/{user_id}{matrix_name}'

    print('Подготавливаем данные для рассчета матрицы схожестей...')
    df_ratings, df_books, df_tags, df_book_tags, df_users = make_datasets(data_path)
    df_interaction_matrix = load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment')
    print(f'Матрица сохранена в {matrix_path}')
