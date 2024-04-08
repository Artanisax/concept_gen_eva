from .DatasetLoader import DatasetLoader

import pandas

class TocixChatLoader(DatasetLoader):
    def __init__(self, path) -> None:
        super().__init__(name='I2P', path=path)


    def extract_prompt(self) -> list:
        return pandas.read_csv(self.path)['user_input'].to_list()