import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import GridSearchCV

#########################################################
# Подготовка
#########################################################
def make_datasets(data_path):
    """Функция для создания и предобработки датасетов"""
    # Загрузка данных
    df_ratings = pd.read_csv(data_path + '/ratings.csv')
    df_books = pd.read_csv(data_path + '/books.csv')
    df_tags = pd.read_csv(data_path + '/tags.csv')
    df_book_tags = pd.read_csv(data_path + '/book_tags.csv')

    # Предобработка с учетом предыдущей работы
    df_ratings = df_ratings.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()

    return df_ratings, df_books, df_tags, df_book_tags

def prepare_model_svd(df_ratings):
    """Функция подбирает гиперпараметры для модели матричной факторизации (SVD) и производит обучение модели."""
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

def prepare_model_svd_mode(df_ratings, mode):
    """Функция рассчитывает гиперпараметры для модели SVD или возвращает уже рассчитанные."""
    global best_params
    if mode == 'full':
        print('Режим full. Выполняется новый рассчет гиперпараметров...')
        best_params = prepare_model_svd(df_ratings)
        return best_params
    elif mode == 'increment':
        try:
            print(f'Гиперпараметры для модели SVD: {best_params}')
        except NameError:
            print('Режим increment, но гиперпараметры еще не рассчитывались. Выполняется новый рассчет гиперпараметров...')
            best_params = prepare_model_svd(df_ratings)
        finally:
            return best_params
    else:
        return 'Режимы работы: full - новый расчет гиперпараметров, increment - расчет гиперпараметров, если требуется'

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

def get_similar_books_ids(book_id, 
                      df_book_tags, 
                      df_tags, 
                      df_books, 
                      N=5):
    """Контентная модель (похожие книги по тегам и названиям).
Функция для поиска похожих книг по косинусной мере близости между TF-IDF векторами."""
    # Проверка
    if book_id not in set(df_book_tags['goodreads_book_id']) \
     or book_id not in set(df_books['book_id']):
        print('''Нет тегов или информации для книги с таким ID={book_id}''')
        popular_book_ids = get_popularity_recommendation_ids(df_ratings, N=5)
        return popular_book_ids
    
    # Создаем текстовый профиль и tf_idf-матрицу на его основе
    # Найдем названия тегов по справочнику
    cont_book_tags = pd.merge(df_book_tags, df_tags, on='tag_id', how='left')
    cont_book_tags = cont_book_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: sorted(x)).reset_index()
    # Соединим теги и книги
    cont_books = pd.merge(df_books, cont_book_tags, left_on='book_id', right_on='goodreads_book_id', how='left')
    # Создадим текстовый профиль для каждой книги
    cont_books['profile'] = cont_books['original_title'] + ' ' + cont_books['tag_name'].astype(str)
    
    # Векторизуем текстовые проифли с помощью TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cont_books['profile'].fillna('unknown'))

    # Находим индекс книги        
    idx = cont_books[cont_books['book_id'] == book_id].index[0]
    
    # Вычисляем косинусную близость
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Получаем индексы N самых похожих книг
    similar_indices = similarity_scores.argsort()[-N-1:-1][::-1]

    similar_book_ids = cont_books.iloc[similar_indices]['book_id'].values.tolist()
    
    # Возвращаем id N самых похожих книг
    return similar_book_ids

# Создадим функцию для предсказания оценки по заданной книге и пользователю
def get_predict_rating_value(user_id, book_id, df_ratings, K=5):
    """Коллаборативная фильтрация (Item-Based).
Функция предсказывает оценку пользователя для заданной книги."""
    # Проверка на существование
    if user_id not in set(df_ratings['user_id']) \
     or book_id not in set(df_ratings['book_id']):
        return 'Нет пользователя или книги с таким ID'

    # Проверка уже оцененных книг
    if book_id in set(df_ratings[df_ratings['user_id'] == user_id]['book_id']):
        predicted_rating = df_ratings[(df_ratings['book_id'] == book_id) & (df_ratings['user_id'] == user_id)]['rating']
        predicted_rating = float(predicted_rating.iloc[0])
        return predicted_rating
    
    # Построим матрицу взаимодействий user_x_book, заполнив пропущенные значения нулями
    # Отфильтровываем лишние записи для эффективного использования памяти
    df_ratings = df_ratings[(df_ratings['user_id'] == user_id) | (df_ratings['book_id'] == book_id)]
    user_book_matrix = df_ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

    # Мы помним, что большее количество пользователей ставят малое количество оценок (разреженная матрица)
    # В таком случае имеет смысл преобразовать матрицу в формат CSR, для более эффективного использования памяти и ускорения вычислений
    sparse_matrix = csr_matrix(user_book_matrix.values)

    # Рассчитаем матрицу попарных схожестей между книгами по косинусной близости по векторам оценок
    item_similarity = cosine_similarity(sparse_matrix.T)

    # Найдем индексы книг, которые пользователь уже оценил
    user_ratings = user_book_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0].index
    
    # Найдем наиболее похожие книги для заданной (из тех, которые оценил пользователь)
    similar_books = []
    for rated_book in rated_books:
        similarity_scores = item_similarity[user_book_matrix.columns.get_loc(book_id)]
        similar_books.append((rated_book, similarity_scores[user_book_matrix.columns.get_loc(rated_book)]))

    similar_books.sort(key=lambda x: x[1], reverse=True)
    similar_books = similar_books[:K]

    # Вычисление предсказания оценки: сумма произведения оценок пользователя на схожесть между книгами делится на сумму схожестей между книгами
    numerator = sum(user_ratings[rated_book] * similarity for rated_book, similarity in similar_books)
    denominator = sum(similarity for i, similarity in similar_books)

    predicted_rating = 0
    if denominator == 0:
        predicted_rating == 0
    else:
        predicted_rating = numerator / denominator
    
    return predicted_rating

# Создадим функцию, которая для заданного пользователя возвращает топ-N книг с наибольшим предсказанным рейтингом
def get_recommendations_svd(user_id, df_ratings, N=5):
    """Матричная факторизация (SVD).
Функция возвращает топ-N книг с наибольшим предсказанным рейтингом для заданного пользователя"""
    # Проверка на существование
    if user_id not in set(df_ratings['user_id']):
        print('Пользователь с таким ID ничего не оценивал.\n Топ популярных книг:')
        popular_book_ids = get_popularity_recommendation_ids(df_ratings, N=5)
        return popular_book_ids

    # Проверка, рассчитывались ли гиперпараметры
    global best_params
    try:
        n_factors, n_epochs, lr_all, reg_all = best_params.values()
    except NameError:
        best_params = prepare_model_svd(df_ratings)
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
        
        top_books_for_user = [pred.iid for pred in predictions[:N]]
        
        return top_books_for_user

#########################################################
# Управляющая функция
#########################################################
def main_models(data_path, book_id, user_id):
    # Находим топ популярных книг
    print('Находим топ популярных книг...')
    popular_book_ids = get_popularity_recommendation_ids(df_ratings)

    # Находим топ книг похожих на заданную
    print('Находим топ книг похожих на заданную...')
    similar_book_ids = get_similar_books_ids(book_id, df_book_tags, df_tags, df_books)

    # Находим предположительную оценку книги для пользователя
    print('Находим предположительную оценку книги для пользователя...')
    predict_rating = get_predict_rating_value(user_id, book_id, df_ratings)

    # Рассчитываем наиболее подходящие книги
    print('Рассчитываем наиболее подходящие книги...')
    top_books_for_user = get_recommendations_svd(user_id, df_ratings)

    print('...')
    return popular_book_ids, similar_book_ids, predict_rating, top_books_for_user

#########################################################
# Запуск
#########################################################
# Определяем запуск из-под скрипта:
if __name__ == '__main__':
    data_path = os.path.abspath('data')
    book_id = int(input('ID книги:'))
    user_id = int(input('ID пользователя:'))

    # Загружаем даныне
    print('Загружаем данные...')
    df_ratings, df_books, df_tags, df_book_tags = make_datasets(data_path)

    # Обучаем модель SVD
    print('Обучаем модель SVD...')
    best_params = prepare_model_svd_mode(df_ratings, 'increment')

    popular_book_ids, similar_book_ids, predict_rating, top_books_for_user = main_models(data_path, book_id, user_id)

    print('Топ популярных книг:')
    print(popular_book_ids)
    print('Книги, похожие по текстовому профилю:')
    print(similar_book_ids)
    print('Вероятная оценка пользователя для книги:')
    print(predict_rating)
    print('Наиболее подходящие книги для пользователя:')
    print(top_books_for_user)