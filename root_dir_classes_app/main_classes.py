from app_classes.utils.MakeDataset import MakeDataset, Config


if __name__ == '__main__':
    # Создаем конфиг
    config = Config()

    # Создаем экземпляр класса MakeDataset
    maker = MakeDataset(config)

    # Создаем и предобрабатываем датасеты
    df_ratings, df_books, df_tags, df_book_tags, df_users = maker.make_datasets()

    # Выводим инфо о полученных датасетах
    print(df_ratings.info())
    print(df_books.info())
    print(df_tags.info())
    print(df_book_tags.info())
    print(df_users.info())