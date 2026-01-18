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


class MatrixCreator:
    def __init__(self, data_path, matrix_folder, matrix_name):
        self.data_path = data_path
        self.matrix_folder = matrix_folder
        self.matrix_name = matrix_name
        self.df_ratings = None
        self.df_books = None
        self.df_tags = None
        self.df_book_tags = None
        self.df_users = None
        self.df_interaction_matrix = None
        self.user_id = None

    ################################################################################################################################################################
#   Подготовка векторизированной матрицы текстовых профилей                                                                                                    #
################################################################################################################################################################
    def create_similarity_matrix(self, df_book_tags, df_tags, df_books, N=5):
        """Создаёт матрицу tf-idf профилей книг."""
        # Создаем текстовый профиль и tf_idf-матрицу на его основе
        # Найдем названия тегов по справочнику
        cont_book_tags = pd.merge(df_book_tags, df_tags, on='tag_id', how='left')
        cont_book_tags = cont_book_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: sorted(x)).reset_index()
        
        # Соединим теги и книги
        # Создадим текстовый профиль для каждой книги
        cont_books = pd.merge(df_books, cont_book_tags, left_on='book_id', right_on='goodreads_book_id', how='left')
        cont_books['profile'] = cont_books['original_title'] + ' ' + cont_books['tag_name'].astype(str)
        books_profile = cont_books[['book_id', 'profile']]
        
        # Векторизуем текстовые профили с помощью TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(books_profile['profile'].fillna('unknown'))

        return books_profile, tfidf_matrix

#################################################################################################################################################################   Подготовка матрицы схожестей
################################################################################################################################################################
    def create_interaction_matrix(self, user_id, mode='increment'):
        """Создаёт матрицу схожестей книг с историей взаимодействия пользователя."""
        if mode == 'increment':
            try:
                self.df_interaction_matrix
            except AttributeError:
                mode = 'full'

        if mode == 'full':
            # Создаем текстовый профиль и TF-IDF-матрицу на его основе
            books_profile, tfidf_matrix = self.create_similarity_matrix(self.df_book_tags, self.df_tags, self.df_books)
            
            # Создаем матрицу схожестей пользователя и книг 
            books = books_profile["book_id"].unique()
            interaction_matrix = np.zeros((1, len(books)))

            # Прогресс-бар в вывод
            with alive_bar(1) as bar:
                # Оценённые книги пользователя
                rated_books = self.df_ratings[self.df_ratings["user_id"] == user_id]["book_id"]
                
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
            
            self.df_interaction_matrix = pd.DataFrame(interaction_matrix, index=[user_id], columns=books)
        
        return self.df_interaction_matrix

    def load_interaction_matrix(self, matrix_path, mode='increment'):
        """Проверяет, существует ли матрица схожести, и возвращает её."""
        try:
            if mode == 'increment':
                self.df_interaction_matrix = pd.read_csv(matrix_path, index_col=0, sep=';')
            elif mode == 'full':
                self.df_interaction_matrix = self.create_interaction_matrix(self.user_id, mode)
                self.df_interaction_matrix.to_csv(matrix_path, index=True, sep=";")
            else:
                raise AttributeError(f'mode={mode}. Режимы работы: full - новый расчет гиперпараметров, increment - расчет гиперпараметров, если требуется')
        except FileNotFoundError:
            self.df_interaction_matrix = self.create_interaction_matrix(self.user_id, mode)
            self.df_interaction_matrix.to_csv(matrix_path, index=True, sep=";")
        except AttributeError:
            raise
        finally:
            return self.df_interaction_matrix

    ######################################################
    # Запуск
    ######################################################
    def run(self):
        load_dotenv()

        self.user_id = int(input('Введите Ваш ID: '))
        
        matrix_path = f"{self.matrix_folder}/{self.user_id}{self.matrix_name}"

        print('Подготавливаем данные для рассчета матрицы схожестей...')
        self.df_ratings, self.df_books, self.df_tags, self.df_book_tags, self.df_users = make_datasets(self.data_path)
        self.df_interaction_matrix = self.load_interaction_matrix(matrix_path, mode='increment')
        print(f'Матрица сохранена в {matrix_path}')

# Пример использования класса
if __name__ == '__main__':
    recommender = RecommenderSystem(
        data_path=os.path.abspath(os.getenv('data_path')),
        matrix_folder=os.path.abspath(os.getenv('matrix_path')),
        matrix_name=os.getenv('matrix_name')
    )
    recommender.run()