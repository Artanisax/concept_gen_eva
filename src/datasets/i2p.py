from .DatasetLoader import DatasetLoader

import pandas

class I2PLoader(DatasetLoader):
    def __init__(self, path) -> None:
        super().__init__(name='I2P', path=path)


    def extract_prompt(self) -> list:
        return pandas.read_csv(self.path)['prompt'].to_list()