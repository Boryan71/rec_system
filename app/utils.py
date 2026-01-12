import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


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

def generate_user_book_similarity_matrix(df_ratings, df_books, df_book_tags, df_tags):
    """Функция создает матрицу схожестей книг с пользовательскими предпочтениями"""
    # Создаем текстовый профиль и tf_idf-матрицу на его основе
    books_profile, tfidf_matrix = create_similarity_matrix(df_book_tags, df_tags, df_books)
    
    # Создаем матрицу схожестей пользователей и книг 
    users = df_ratings["user_id"].unique()
    books = books_profile["book_id"].unique()
    interaction_matrix = np.zeros((len(users), len(books)))
    
    # Оцениваем схожесть с историей пользователя
    for user_id in users:
        # Оценённые книги пользователя
        rated_books = df_ratings[df_ratings["user_id"] == user_id]["book_id"]
        
        # Пропускаем, если нет оценок
        if len(rated_books) == 0:
            continue
        
        # Пропускаем, если у оцененных нет тегов
        rated_indices = books_profile[books_profile["book_id"].isin(rated_books)].index
        if len(rated_indices) == 0:
            continue
        
        # Находим и нормируем косинусное сходство оцененных и неоцененных книг
        rated_tfidf = tfidf_matrix[rated_indices]
        similarity = cosine_similarity(np.asarray(rated_tfidf.mean(axis=0)).reshape(1,-1), tfidf_matrix).flatten()
        scaled_similarity = similarity * 100
        
        # Записываем в матрицу
        interaction_matrix[np.where(users == user_id)[0][0], :] = scaled_similarity
    
    df_interaction = pd.DataFrame(interaction_matrix, index=users, columns=books)
    
    return df_interaction

def make_datasets(data_path):
    """Функция для создания и предобработки датасетов."""    
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
# Подготовка моделей
#########################################################
def create_similarity_matrix(df_book_tags, df_tags, df_books, N=5):
    """Функция создает матрицу tf-idf профилей книг."""
    # Создаем текстовый профиль и tf_idf-матрицу на его основе
    # Найдем названия тегов по справочнику
    cont_book_tags = pd.merge(df_book_tags, df_tags, on='tag_id', how='left')
    cont_book_tags = cont_book_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: sorted(x)).reset_index()
    # Соединим теги и книги
    cont_books = pd.merge(df_books, cont_book_tags, left_on='book_id', right_on='goodreads_book_id', how='left')
    # Создадим текстовый профиль для каждой книги
    cont_books['profile'] = cont_books['original_title'] + ' ' + cont_books['tag_name'].astype(str)
    books_profile = cont_books[['book_id', 'profile']]
    
    # Векторизуем текстовые проифли с помощью TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(books_profile['profile'].fillna('unknown'))

    return books_profile, tfidf_matrix

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