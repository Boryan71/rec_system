import pandas as pd
from ..config.Config import Config


class MakeDataset:
    """Класс отвечает за загрузку и предварительную обработку датасетов"""

    def __init__(self, data_path):
        self.data_path = data_path

    def add_user_features(self, df_ratings):
        """Функция рассчитывает расширенные признаки пользователей.
        :param df_ratings:    DataFrame с рейтингами

        :return:    DataFrame с пользовательскими атрибутами"""
        # Средняя оценка пользователя
        user_avg_rating = (
            df_ratings.groupby("user_id")["rating"].mean().rename("avg_user_rating")
        )
        # Количество оценок пользователя
        user_num_ratings = (
            df_ratings.groupby("user_id")["rating"].count().rename("num_user_ratings")
        )
        # Активность пользователя
        total_unique_books = len(df_ratings["book_id"].unique())
        user_activity = user_num_ratings / total_unique_books * 100
        user_activity.name = "user_activity"

        df_users = pd.concat([user_avg_rating, user_num_ratings, user_activity], axis=1)
        df_users = df_users.reset_index()
        return df_users

    def add_book_features(self, df_ratings, df_books, df_book_tags):
        """Функция рассчитывает расширенные признаки книг.
        :param df_ratings:    DataFrame с рейтингами
        :param df_books:      DataFrame с книгами

        :return: DataFrame с расширенными атрибутами книг"""
        # Популярность книги
        book_popularity = (
            df_ratings.groupby("book_id")["user_id"]
            .nunique()
            .rename("popularity")
            .to_frame()
        )
        # Разнообразие оценок (стандартное отклонение)
        book_rating_std = (
            df_ratings.groupby("book_id")["rating"]
            .std()
            .rename("rating_std")
            .to_frame()
        )
        # Самый популярный тег
        top_tag_per_book = df_book_tags.groupby("goodreads_book_id")["count"].idxmax()
        df_top_tags = df_book_tags.iloc[top_tag_per_book].copy()
        df_top_tags.rename(
            columns={"tag_id": "top_tag", "goodreads_book_id": "book_id"}, inplace=True
        )
        df_top_tags.drop(columns=["count"], inplace=True)

        # Добавляем признаки
        df_books_with_tags = df_books.merge(df_top_tags, on="book_id", how="left")
        df_books_extended = df_books_with_tags.merge(
            book_popularity, on="book_id", how="left"
        )
        df_books_extended = df_books_extended.merge(
            book_rating_std, on="book_id", how="left"
        )

        # Оставляем только существенные признаки
        df_books_extended = df_books_extended[
            [
                "id",
                "book_id",
                "best_book_id",
                "authors",
                "original_title",
                "title",
                "language_code",
                "average_rating",
                "ratings_count",
                "work_ratings_count",
                "work_text_reviews_count",
                "top_tag",
                "popularity",
                "rating_std",
            ]
        ]

        return df_books_extended

    def make_datasets(self):
        """Функция для создания и предобработки датасетов."""

        # Загрузка данных
        df_ratings = pd.read_csv(self.data_path + "/ratings.csv")
        df_books = pd.read_csv(self.data_path + "/books.csv")
        df_tags = pd.read_csv(self.data_path + "/tags.csv")
        df_book_tags = pd.read_csv(self.data_path + "/book_tags.csv")

        # Предобработка с учетом EDA
        df_ratings = (
            df_ratings.groupby(["user_id", "book_id"])["rating"].mean().reset_index()
        )

        # Создание расширенных признаков
        df_users = self.add_user_features(df_ratings)
        df_books = self.add_book_features(df_ratings, df_books, df_book_tags)

        self.df_ratings = df_ratings
        self.df_books = df_books
        self.df_tags = df_tags
        self.df_book_tags = df_book_tags
        self.df_users = df_users

        return df_ratings, df_books, df_tags, df_book_tags, df_users


if __name__ == "__main__":
    conf = Config()
    data_path = conf.data_path
    md = MakeDataset(data_path)
    
    df_ratings, df_books, df_tags, df_book_tags, df_users = md.make_datasets()

    dataset_list = [df_ratings, df_books, df_tags, df_book_tags, df_users]
    for i in dataset_list:
        print(i.info())
