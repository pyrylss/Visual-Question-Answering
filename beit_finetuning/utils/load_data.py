import numpy as np
import glob, json, pickle, random
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer

from .ans_punct import prep_ans
import torchvision.transforms as transforms
from PIL import Image
import os
import csv

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transformers import XLMRobertaTokenizer

def soft_target(answers, ans_to_ix, preprocess=True):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    for ans in answers:
        if preprocess:
            ans = prep_ans(ans)
        if ans in ans_to_ix:
            ans_score[ans_to_ix[ans]] = min(1.0, ans_score[ans_to_ix[ans]] + 0.3)
    return ans_score


class CommonData:
    """
    load common data for all dataset objects:
    * imgid_to_path
    * bert tokenizer
    * ans_to_ix, ix_to_ans
    """
    def __init__(self, __C) -> None:
        print('Loading common data...')

        self.img_path_list = []
        for split in __C.FEATURE_SPLIT:
            feats_dir = __C.IMAGE_DIR[split]
            self.img_path_list += glob.glob(feats_dir + '*.jpg')
        self.imgid_to_path = {}
        for ip in self.img_path_list:
            img_id = int(ip.split('/')[-1].split('_')[-1].split('.')[0])
            self.imgid_to_path[img_id] = ip

        print(f'== Total image number: {len(self.imgid_to_path)}')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        spm_path = os.path.join(current_dir, "beit3.spm")

        # load bert tokenizer
        self.tokenizer = XLMRobertaTokenizer(spm_path)

        self.token_size = self.tokenizer.vocab_size
        print(f'== BertTokenizer loaded, vocab size: {self.token_size}')

        # load ans_to_ix, ix_to_ans
        ans_dict_path = __C.ANSWER_DICT_PATH[__C.DATA_TAG]
        self.ix_to_ans = json.load(open(ans_dict_path, 'r'))
        self.ans_to_ix = {ans: ix for ix, ans in enumerate(self.ix_to_ans)}
        self.ans_size = len(self.ans_to_ix)
        print(f'== Answer vocab size: {self.ans_size}')

        print('Common data process is done.\n')
        

class DataSet(Data.Dataset):
    def __init__(self, __C, common_data, split_name_list):
        self.__C = __C
           
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

        print(f'Loading dataset for {self.__C.TASK}|{self.__C.RUN_MODE}({split_name_list})')
        self.split_name_list = split_name_list

        # load all attributes from common data
        self.imgid_to_path = common_data.imgid_to_path
        self.tokenizer = common_data.tokenizer
        self.token_size = common_data.token_size
        self.ans_to_ix = common_data.ans_to_ix
        self.ix_to_ans = common_data.ix_to_ans
        self.ans_size = common_data.ans_size

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        for split_name in split_name_list:
            ques_list = json.load(open(__C.QUESTION_PATH[split_name], 'r'))
            if 'questions' in ques_list:
                ques_list = ques_list['questions']
            self.ques_list += ques_list
            if split_name in __C.ANSWER_PATH:
                ans_list = json.load(open(__C.ANSWER_PATH[split_name], 'r'))
                if 'annotations' in ans_list:
                    ans_list = ans_list['annotations']
                self.ans_list += ans_list

        # indexing data, note that all question_id is set to str,
        # and all image_id is set to int
        if len(self.ans_list) == len(self.ques_list):
            self.annotated = True
            self.qids = [str(ans['question_id']) for ans in self.ans_list]
        elif len(self.ans_list) < len(self.ques_list):
            self.annotated = False
            self.qids = [str(ques['question_id']) for ques in self.ques_list]
        else:
            raise ValueError('Answer list is longer than question list!')

        self.data_size = len(self.qids)
        print(f'== data size: {self.data_size}\n')

        self.qid_to_ques = {str(ques['question_id']): ques for ques in self.ques_list}
        self.qid_to_ans = {str(ans['question_id']): ans for ans in self.ans_list}

    def __getitem__(self, idx):
        # get question in token ids, image in features,
        # and answer in binary-label vector

        __C = self.__C

        # For code safety
        img_feat  = np.zeros(1)
        ques_ids  = np.zeros(1)
        ans_vec   = np.zeros(1)

        qid = self.qids[idx]
        ques_info = self.qid_to_ques[qid]
        
        # Process question
        ques_str = ques_info['question']
        ques_ids = self.beit3_tokenize(ques_str, __C.MAX_TOKEN)
       
        # Process image feature
        img_id = int(ques_info['image_id'])
        
        img_path = self.imgid_to_path[img_id]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.annotated:
            ans_info = self.qid_to_ans[qid]
            if 'answers' in ans_info:
                ans_list = [ans['answer'] for ans in ans_info['answers']]
            elif 'direct_answers' in ans_info:
                ans_list = ans_info['direct_answers']
            else:
                raise ValueError('Error: annotation format is not supported!')
            assert type(ans_list[0]) == str, 'Error: answer format is not supported!'
            ans_vec = soft_target(ans_list, self.ans_to_ix)
        return image, torch.tensor(ques_ids, dtype=torch.long), torch.tensor(ans_vec, dtype=torch.float)


    def __len__(self):
        return self.data_size

    def beit3_tokenize(self, text, max_token):
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_token)
        padding_length = max_token - len(tokens)
        tokens += [self.tokenizer.pad_token_id] * padding_length
        padding_mask = [0] * len(tokens) + [1] * padding_length
        return tokens#, padding_mask
