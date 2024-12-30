import json
import pickle

class IngConfg:

    _vocabs = ["ingr_only", "ingr_title", "ingr_tag", "ingr_title_tag"]

    def __init__(
            self, vocab="ingr_title", path="/home/donghee/projects/mlm2/config.json"
    ):
        self.path = path
        self.vocab = vocab
        with open(self.path, 'r') as f:
            self.static_config = json.load(f)
        
        # data config
        self.data_config = self.static_config['data_folder']

        self.original_data_folder = f"{self.data_config['home_dir']}/{self.data_config['original_dir']}"

        # bert config
        self.bert_config = self.static_config["bert_config"]

        self.bert_resize_embedding = self.bert_config["resize_embedding"]

        if vocab in self._vocabs:
            self.vocab_path = self.bert_config["tokenizer"]["tokenizer_file_path"][vocab]
        else:
            raise ValueError(f"vocab parameter is not valid: {vocab}")
    
    def get_dictionary(self):
        with open(f"{self.original_data_folder}/iid2ingr_full.pkl", 'rb') as f:
            self.iid2ingr_data = pickle.load(f)
        self.ing_list = list(self.iid2ingr_data.values())

    def __str__(self):
        return 
    
    def __repr(self):
        return self.__str__()
    
