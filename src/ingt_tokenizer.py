import bootstrap

from datasets import load_dataset
from tokenizers import (Tokenizer, models, normalizers,
                        pre_tokenizers)
from transformers import PreTrainedTokenizerFast

class IngTokenizer:
    def __init__(self, ingt_config):
        self.ingt_config = ingt_config.bert_config["tokenizer"]
        self.vocab = ingt_config.vocab

    def __str__(self) -> str:
        return f"Custom Tokenizer: {self.ingt_config}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _get_training_corpus(self, ds, split='train', batch_size= 100):
        _current_ds = ds[split]
        for i in range(0, len(_current_ds), batch_size):
            yield _current_ds[i: i+batch_size]["text"]

    def train(self):
        self._tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        self._tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

        self._tokenizer.pre_tokenizer = pre_tokenizers.WhitesplaceSplit()

        _train_file_path = self.ingt_config["tokenizer_train_file_path"][self.vocab]

        self.ingt_ds = load_dataset("text", data_files={"train":_train_file_path})

        special_tokens = self.ingt_config["special_tokens"]

        trainer = trainer.WordPieceTrainer(vocab_size=self.ingt_config["vocab_max_size"], special_tokens=special_tokens)

        self._tokenizer.train_from_iterator(self._get_training_corpus(self.ingt_ds), trainer=trainer)

    def save(self):
        if self._tokenizer is None:
            raise ValueError("Tokenizer is None, maybe train first")
        self._tokenizer.save(self.ingt_config["tokenizer_file_path"][self.vocab])

    def load(self):
        self.tokenzier = PreTrainedTokenizerFast(
            tokenizer_file=self.ingt_config["tokenizer_file_path"][self.vocab],
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            max_len=self.ingt_config["max_len"],
        )