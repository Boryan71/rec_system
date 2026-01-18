import os
from dotenv import load_dotenv


class Config:
    """Класс отвечает за передачу переменных между модулями"""

    def __init__(self):
        load_dotenv()
        self.data_path = os.path.abspath(os.getenv("data_path"))
        self.matrix_folder = os.path.abspath(os.getenv("matrix_path"))
        self.matrix_name = os.getenv("matrix_name")
        self.params_path = os.path.abspath(os.getenv("params_path"))


if __name__ == "__main__":
    conf = Config()
    print(conf.data_path)
    print(conf.matrix_folder)
    print(conf.matrix_name)
    print(conf.params_path)
