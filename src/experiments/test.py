import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import argparse
import logging
import wandb
import pandas as pd
from tqdm import tqdm
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                          AutoConfig, AutoModelForMaskedLM, )

import bootstrap
from ingt_tokenizer import IngTokenizer

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed()

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(0)

def arg_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, default='', help='wandb project name')
    argparser.add_argument('--data_path', type=str, default='', help='dataset path')
    argparser.add_argument('--mdeol_name_or_checkpoint_path', type=str, default='', help='model name or checkpoint path')
    argparser.add_argument('--mask_position', type=str, default='', nargs=6, help='where to mask: first, last, random, top_0_33p, top_33p_66p, top_66p_100p')
    return argparser.parse_args()

args = arg_parse()

VOCAB_CONFIG = 'ingr_ony'
CONFIG_PATH = "/home/donghee/projects/mlm2/config.json"
ing_config = bootstrap.IngConfig(vacab=VOCAB_CONFIG, path=CONFIG_PATH)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

data_folder = args.data_path
model_name_or_path = args.model_name_or_checkpoint_path

ing_tokenizer = IngTokenizer(ing_config)
ing_tokenizer.load()
tokenizer = ing_tokenizer.tokenizer

config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(
    model_name_or_path,
    config=config,
)

def split_list(li):
    div3 = len(li)//3
    if len(li) % 3 == 2:
        return li[:div3+1], li[div3+1:-div3], li[-div3:]
    return li[:div3], li[div3:-div3], li[-div3:]

def inference_and_mk_tsv(ver):
    line_list = []; q_list = []; answer_list =[]; pred_list = []

    with open(f"/media/ssd/dh/projects/ing_mlm/data_folder/processed/v1_ing_only/test.txt", 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) == 0 :
                continue
            

            line_list.append(line)
            word_list = line.split()  
            if ver == 'first' : # 맨 앞 [MASK]
                answer_idx = 0
                answer = word_list[answer_idx]
            elif ver == 'last' : # 맨 뒤 [MASK]
                answer_idx = -1
                answer = word_list[answer_idx]
            elif ver == 'random' : # RANDOM [MASK]
                answer_idx = random.choice(range(len(word_list)))
                answer = word_list[answer_idx]
                
            elif ver == 'top_0_33p' : # 구간 상위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 0 
                answer = random.choice(word_list_chuncked[answer_idx])
            elif ver == 'top_33p_66p' : # 구간 중위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 1
                answer = random.choice(word_list_chuncked[answer_idx])
            elif ver == 'top_66p_100p' : # 구간 하위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 2 
                answer = random.choice(word_list_chuncked[answer_idx])
            q = line.replace(answer, "[MASK]")
            q_list.append(q)
            answer_list.append(answer)
            
    df = pd.DataFrame()
    df['original'] = line_list
    df['q'] = q_list
    df['answer'] = answer_list
    df['ori_sp'] = df['original'].apply(lambda x : x.split(' '))
    df['ori_len'] = df['ori_sp'].apply(lambda x : len(x))
    df['q_tokenized'] = df['q'].apply(lambda x : tokenizer(str(x),  return_tensors="pt"))

    pred_list = []

    _topk_num = 10
    _pred_topk_list = [[] for i in range(_topk_num)]
    _device = torch.device('cuda:1')

    model.to(_device)

    for _tokenized in tqdm(df['q_tokenized']):
        _t = {k:v.to(_device)for k,v in _tokenized.items()}
    
        _pred = model(**_t)
        _logit = _pred.logits[0][0].detach().cpu()
        del _pred.logits
        del _pred
        del _t
        del _tokenized
        torch.cuda.empty_cache()
        pred_list.append(_logit)
    
        _res = tokenizer.convert_ids_to_tokens(torch.topk(_logit, _topk_num).indices)
        for _i, _token in enumerate(_res):
            _pred_topk_list[_i].append(_token)
        
    for _i, _pred_tokens in enumerate(_pred_topk_list):
        df[f'top_{_i}'] = _pred_tokens
        
    return df

def inference_score(dataframe_) : 
    dataframe_['errata_1'] = " "
    dataframe_['errata_3'] = " "
    dataframe_['errata_5'] = " "
    dataframe_['errata_10'] = " "
    
    for k in [1,3,5,10] : 
        for idx, value in enumerate(dataframe_.index) : 
            #print(dataframe_['answer'][idx])
            topk = list(dataframe_.iloc[idx, 6:k+6]) 
            #print(topk)
            if dataframe_['answer'][idx] in topk : 
                dataframe_['errata_' + str(k)][idx] = 1 
                
            else :
                dataframe_['errata_' + str(k)][idx] = 0 
                
    
    length = len(dataframe_)
    
    acc_1 = dataframe_['errata_1'].value_counts()[1] / length
    acc_3 = dataframe_['errata_3'].value_counts()[1] / length
    acc_5 = dataframe_['errata_5'].value_counts()[1] / length
    acc_10 = dataframe_['errata_10'].value_counts()[1] / length
    
    return acc_1, acc_3, acc_5, acc_10

def main():

    for mp in args.mask_position:
        df = inference_and_mk_tsv(mp)

        acc_1, acc_3, acc_5, acc_10 = inference_score(df)
        print(f'acc_1: {acc_1}, acc_3: {acc_3}, acc_5: {acc_5}, acc_10: {acc_10}')

        m_last_path = os.path.basename(os.path.normpath(model_name_or_path))
        m_last_path_ckpt_num = m_last_path.split('-')

        run = wandb.init(project='test_exp')
        wandb.config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 128,
        }
        run.name = f"{m_last_path_ckpt_num[0]} - {int(m_last_path_ckpt_num[1]) :07d} - {mp}"
        run.log({'acc@1': acc_1, 'acc@3': acc_3, 'acc@5': acc_5, 'acc@10': acc_10})

        save_path = '/home/donghee/projects/mlm2/src/experiments/results/'

        df.to_csv(f'{save_path}/test_2023_0324/{m_last_path}-{mp}.tsv', index=False, sep="\t") 

        #artifact
        with wandb.init(project="test_exp", dir=f"/media/ssd/dh/projects/ing_mlm/processing/test_2023_0324/{m_last_path}", job_type="load-data") as run: 

            # 🏺 create our Artifact
            exp_artifact = wandb.Artifact(
                "exp_artifact-data", type="dataset",
                description="test")
        
            # 🐣 Store a new file in the artifact, and write something into its contents.
            exp_artifact.add_file(f'{save_path}/test_2023_0324/{m_last_path}-{mp}.tsv') 
            
            # ✍️ Save the artifact to W&B.
            run.log_artifact(exp_artifact)

if __name__ == "__main__":
    print('start performance verification')
    main()