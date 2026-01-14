import os
import sys
import random
import pandas as pd
from dotenv import load_dotenv
import hybrid_system as hs
import individual_models as im
from util_make_datasets import *
from util_load_best_params_svd import *
from util_load_interaction_matrix import *

load_dotenv()

# Переменные окружения
DATA_PATH = os.environ.get('data_path')
params_path = im.params_path = os.environ.get('params_path')
matrix_folder = os.environ.get('matrix_path')
matrix_name = os.environ.get('matrix_name')

# Загрузка данных
df_ratings, df_books, df_tags, df_book_tags, df_users = make_datasets(DATA_PATH)

# Управление режимом работы
MODE = input('Выберите режим работы (full/increment): ')
user_id = int(input('\nВведите Ваш ID пользователя: '))
user_type = hs.get_user_type(user_id, df_users)
matrix_path = f'{matrix_folder}/{user_id}{matrix_name}'
print(f'\nТип пользователя: {user_type}\n')

if MODE.lower() == 'full':
    best_params = load_best_params_svd(params_path, df_ratings, mode='full')
    hs.df_interaction_matrix = load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='full')
elif MODE.lower() == 'increment':
    best_params = load_best_params_svd(params_path, df_ratings, mode='increment')
    hs.df_interaction_matrix = load_interaction_matrix(matrix_path, user_id, df_ratings, df_books, df_book_tags, df_tags, mode='increment')
else:
    print('Неподдерживаемый режим.')
    sys.exit()

# Общие списки книг
recommended_editorial = hs.get_combined_recomendation_by_user_type(user_id, df_ratings, df_books, df_tags, df_book_tags, df_users)
popular_books = im.get_popularity_recommendation_ids(df_ratings, N=5)

# Функционал главной страницы
def home_page():
    print('|----------------------|')
    print('| Главная страница     |')
    print('|----------------------|')
    print('| Популярное.......... |')
    print('|----------------------|')
    display_books(im.get_popularity_recommendation_ids(df_ratings, N=5))
    print('| Может понравиться.... |')
    print('|----------------------|')
    display_books(im.get_similar_books_ids(df_book_tags, df_tags, df_books, df_ratings, random.choice(popular_books), N=10))
    print('| Рекомендации......... |')
    print('|----------------------|')
    display_books(recommended_editorial[:10])

# Показ книг
def display_books(books):
    for book_id in books:
        try:
            book_row = df_books[df_books['id'] == book_id].iloc[0]
            print(f'- {book_row["id"], book_row["original_title"]}, {book_row["authors"]}, ({book_row["average_rating"]:.2f})')
        except Exception:
            continue

# Просмотр страницы книги
def view_book_details(book_id):
    book_row = df_books[df_books['book_id'] == book_id].iloc[0]
    print('|-------------------------|')
    print(f'| Книга: {book_row["original_title"]} |')
    print('|-------------------------|')
    print(f'Автор: {book_row["authors"]}')
    print(f'Год издания: {book_row["publication_year"]}')
    print(f'Средняя оценка: {book_row["average_rating"]:.2f}')
    print(f'Вероятная оценка пользователя: {get_predict_rating_value(user_id, book_id, df_ratings)}')
    print('|-------------------------|')
    print('| Похожие книги......     |')
    print('|-------------------------|')
    similar_books = get_similar_books_ids(df_book_tags, df_tags, df_books, df_ratings, book_id, N=10)
    display_books(similar_books)
    print('|-------------------------|')
    print('| Рекомендуемые кандидаты |')
    print('|-------------------------|')
    candidates = hs.get_candidate_pool(user_id, df_ratings, df_books, df_tags, df_book_tags, book_id)
    filtered_candidates = hs.set_diversity_filter(candidates, df_books)
    display_books(filtered_candidates)

# Начало программы
while True:

    home_page()

    choice = input("\nХотите перейти на страницу книги? (Y/N): ")
    if choice.upper() == 'Y':
        book_id = int(input('Введите ID книги для просмотра подробностей: '))
        view_book_details(book_id)
        back_choice = input("\nВернуться на главную страницу? (Y/N): ")
        if back_choice.upper() == 'N':
            break
    else:
        break