import json

class IngConfg:

    _vocabs = ["ingr_only", "ingr_title", "ingr_title_tag"]

    def __init__(
            self, vocab="ingr_title", path="/home/donghee/projects/mlm2/config.json"
    ):
        self.path = path
        self.vocab = vocab
        with open(self.path, 'r') as f:
            self.static_config = json.load(f)
        
        self.data_config = self.static_config['data_folder']

        self.original_data_folder = f"{self.data_config['home_dir']}/{self.data_config['original_dir']}"



    def __str__(self):
        return 
    
    def __repr(self):
        return self.__str__()
    
