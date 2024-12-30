import bootstrap
from ingt_tokenizer import IngtTokenizer
from loguru import logger

file_path = "/home/donghee/projects/mlm/config.json" 

# logger.info(' >> ingr only')
# ingt_config = bootstrap.IngConfig(vocab='ingr_only',path=file_path)
# ingt_tokenizer = IngtTokenizer(ingt_config) 
# ingt_tokenizer.train()
# ingt_tokenizer.save()

# # logger.info(' >> ingr title ')
# ingt_config = bootstrap.IngConfig(vocab='ingr_title',path=file_path)
# ingt_tokenizer = IngtTokenizer(ingt_config) 
# ingt_tokenizer.train()
# ingt_tokenizer.save()

# # logger.info(' >> ingr tag ')
# ingt_config = bootstrap.IngConfig(vocab='ingr_tag',path=file_path)
# ingt_tokenizer = IngtTokenizer(ingt_config) 
# ingt_tokenizer.train()
# ingt_tokenizer.save()

# logger.info(' >> ingr title tag ')
# ingt_config = bootstrap.IngConfig(vocab='ingr_title_tag', path=file_path)
# ingt_tokenizer = IngtTokenizer(ingt_config)
# ingt_tokenizer.train()
# ingt_tokenizer.save()