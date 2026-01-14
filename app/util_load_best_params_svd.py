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
import ast
from dotenv import load_dotenv
from util_make_datasets import make_datasets


#########################################################
# Подготовка модели SVD
#########################################################
def prepare_model_svd(df_ratings):
    """Функция подбирает гиперпараметры для модели матричной факторизации (SVD)."""
    # Загрузим данные в формат, подходящий для scikit-surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[['user_id', 'book_id', 'rating']], reader)

    # Гиперпараметры для подбора (Оставляю основные после нескольких подборов)
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [20, 50], 
        'lr_all': [0.005, 0.01],
        'reg_all': [ 0.1]
    }

    # Объект подбора гиперпараметров
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv = 3, n_jobs = -1, joblib_verbose = 101)

    # Выполняем подбор гиперпараметров
    gs.fit(data)
    best_params_rmse = gs.best_params['rmse']

    return best_params_rmse

def prepare_model_svd_mode(df_ratings, mode='increment'):
    """Функция рассчитывает гиперпараметры для модели SVD или возвращает уже рассчитанные."""
    global best_params

    try:
        if mode == 'full':
            print('Режим full. Выполняется новый рассчет гиперпараметров...')
            best_params = prepare_model_svd(df_ratings)
        elif mode == 'increment':
            print(f'Гиперпараметры для модели SVD: {best_params}')
    except NameError:
        print('Режим increment, но гиперпараметры еще не рассчитывались. Выполняется новый рассчет гиперпараметров...')
        best_params = prepare_model_svd(df_ratings)
    finally:
        return best_params

def load_best_params_svd(params_path, df_ratings, mode='increment'):
    """Функция проверяет, рассчитывались ли параметры для модели SVD, и загружает их."""
    try:
        if mode == 'increment':
            with open(params_path, 'r') as f:
                loaded_str = f.read()
                best_params = ast.literal_eval(loaded_str)
        elif mode == 'full':
            best_params = prepare_model_svd_mode(df_ratings, mode)
            with open(params_path, 'w') as f:
                f.write(str(best_params))
        else:
            raise AttributeError(f'mode={mode}. Режимы работы: full - новый расчет гиперпараметров, increment - расчет гиперпараметров, если требуется')
    except FileNotFoundError:
        best_params = prepare_model_svd_mode(df_ratings, mode)
        with open(params_path, 'w') as f:
            f.write(str(best_params))
    except AttributeError:
        raise
    finally:
        return best_params

#########################################################
# Запуск
#########################################################
if __name__ == '__main__':
    load_dotenv()
    data_path = os.path.abspath(os.getenv('data_path'))
    params_path = os.path.abspath(os.getenv('params_path'))

    print('Подготавливаем данные для рассчета...')
    df_ratings, df_books, df_tags, df_book_tags, df_users = make_datasets(data_path)
    best_params = load_best_params_svd(params_path, df_ratings, mode='increment')
    print(f'Параметры сохранены в {params_path}')
