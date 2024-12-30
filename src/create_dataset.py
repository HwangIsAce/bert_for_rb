import bootstrap
from utils import ProcessHandler

from loguru import logger
import pickle
import json
from tqdm import tqdm

ingt_config = bootstrap.IngConfig(
    vocab="ingr_only", path="/home/donghee/projects/mlm2/config.json"
)

data_config = ingt_config.data_config

original_file_path = ingt_config.original_data_folder

logger.info(f"original_file_path: {original_file_path}")
logger.info(f"data_config: {data_config}")

with open(f"{original_file_path}/iids_full.pkl", 'rb') as f:
    iids_data = pickle.load(f)

with open(f"{original_file_path}/iid2ingr_full.pkl", 'rb') as f:
    iid2ingr_data = pickle.load(f)
    
with open(f"{original_file_path}/iid2fgvec_full.pkl", 'rb') as f:
    iid2fgvec_data = pickle.load(f)

with open(f"{original_file_path}/rid2title.pkl", 'rb') as f:
    rid2title_data = pickle.load(f)

with open(f"{original_file_path}/rid2info_full.pkl", 'rb') as f:
    rid2info_full = pickle.load(f)

with open(f"{original_file_path}/rid2sorted_iids_full.pkl", 'rb') as f:
    rid2sorted_iids_data = pickle.load(f)

with open(f"{original_file_path}/tags.json", "rb") as f:
    rid2tags = json.load(f)

_target_dir = f"{data_config['home_dir']}/{data_config['processed_dir']}"
logger.info(f"target_dir: {_target_dir}")

_v1_ing_only_path = f"{_target_dir}/v1_ing_only"
logger.info(f"_v1_ing_only_path = {_v1_ing_only_path}")
v1_handler = ProcessHandler(
    _v1_ing_only_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="v1_ingr_only",
)

_v1_ing_only_sample_path = f"{_target_dir}/v1_ing_only_sample"
logger.info(f"_v1_ing_only_sample_path = {_v1_ing_only_sample_path}")
v1_sample_handler = ProcessHandler(
    _v1_ing_only_sample_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="v1_ingr_only_sample",
)

_v2_ing_title_path = f"{_target_dir}/v2_ing_title"
logger.info(f"_v2_ing_title_path = {_v2_ing_title_path }")
v2_handler = ProcessHandler(
    _v2_ing_title_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="v2_ingr_title",
)

_v2_ing_title_sample_path = f"{_target_dir}/v2_ing_title_sample"
logger.info(f"_v2_ing_title_sample_path = {_v2_ing_title_sample_path }")
v2_sample_handler = ProcessHandler(
    _v2_ing_title_sample_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="v2_ingr_title_sample",
)

_v3_ing_title_tag_path = f"{_target_dir}/v3_ing_title_tag"
logger.info(f"_v3_ing_title_tag_path = {_v3_ing_title_tag_path }")
v3_handler = ProcessHandler(
    _v3_ing_title_tag_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    rid2tags, #js
    ver="v3_ingr_title_tag",
)

_v3_ing_title_tag_sample_path = f"{_target_dir}/v3_ing_title_tag_sample"
logger.info(f"_v3_ing_title_tag_sample_path = {_v3_ing_title_tag_sample_path }")
v3_sample_handler = ProcessHandler(
    _v3_ing_title_tag_sample_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    rid2tags, #js
    ver="v3_ingr_title_tag_sample",
)

_v4_ing_tag_path = f"{_target_dir}/v4_ing_tag"
logger.info(f"_v4_ing_tag_path= {_v4_ing_tag_path}")
v4_handler = ProcessHandler(
    _v4_ing_tag_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    rid2tags,
    ver="v4_ingr_tag",
)

_v4_ing_tag_sample_path = f"{_target_dir}/v4_ing_tag_sample"
logger.info(f"_v4_ing_tag_sample_path= {_v4_ing_tag_sample_path}")
v4_sample_handler = ProcessHandler(
    _v4_ing_tag_sample_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    rid2tags,
    ver="v4_ingr_tag_sample",
)

error_rid_f = open(f"{_target_dir}/error_rids.txt", "w")

for idx, (rid, title) in tqdm(enumerate(rid2title_data.items())):

    if rid not in rid2sorted_iids_data:
        error_rid_f.write(f"{rid}\n")
        continue
    
    v1_handler.write(rid)
    v1_sample_handler.write(rid)

    v2_handler.write(rid)
    v2_sample_handler.write(rid)

    v3_handler.write(rid)
    v3_sample_handler.write(rid)

    v4_handler.write(rid)
    v4_sample_handler.write(rid)

v1_handler.close()
v1_sample_handler.close()
v2_handler.close()
v2_sample_handler.close()
v3_handler.close()
v3_sample_handler.close()
v4_handler.close()
v4_sample_handler.close()

error_rid_f.close()
