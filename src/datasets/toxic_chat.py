from .DatasetLoader import DatasetLoader

import pandas

class ToxicChatLoader(DatasetLoader):
    def __init__(self, path) -> None:
        super().__init__(name='Toxic', path=path)

    # type is in ['user_input', 'model_output']
    def extract_prompt(self, type) -> list:
        df = pandas.read_csv(self.path)
        if type == 'user_input':
            res = df[df['toxicity'] == 1]
        elif type == 'model_output':
            res = df[df['jailbreaking'] == 1]
        return res[type].to_list()