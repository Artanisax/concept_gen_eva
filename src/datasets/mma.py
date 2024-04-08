from .DatasetLoader import DatasetLoader

import pandas

class MMALoader(DatasetLoader):
    def __init__(self, path) -> None:
        super().__init__(name='I2P', path=path)

    
    # type is in ['target', 'adv', 'sanitized_adv', 'clean']
    def extract_prompt(self, type='adv') -> list:
        return pandas.read_csv(self.path)[f'{type}_prompt'].to_list()