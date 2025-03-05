import os, sys
# sys.path.append(os.getcwd())

from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as Data
import argparse
from pathlib import Path
import yaml
from copy import deepcopy
from tqdm import tqdm

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.beit_for_finetune import BEiT3ForFinetune
from .utils.optim import get_optim_for_finetune as get_optim

from torchscale.architecture.config import EncoderConfig


class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C
        self.net = None

    # heuristics generation
    @torch.no_grad()
    def eval(self, dataset):
        data_size = dataset.data_size

        if self.net is None:
            # Load parameters
            path = self.__C.CKPT_PATH
            print('Loading ckpt {}'.format(path))
            
            args2 = _get_large_config(img_size=480)
            args2.normalize_output = False

            net = BEiT3ForFinetune(args2, dataset.ans_size)
            
            ckpt = torch.load(path, map_location='cpu')
            net.load_state_dict(ckpt['state_dict'], strict=False)
            net.cuda()
            if self.__C.N_GPU > 1:
                net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)
            print('Finish!')
            self.net = net
        else:
            net = self.net


        net.eval()
        
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        qid_idx = 0
        latent_results = []

        for step, input_tuple in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            input_tuple = [x.cuda() for x in input_tuple]


            pred, answer_latents = net(input_tuple[:-1], output_answer_latent=True)
            pred_np = pred.sigmoid().cpu().numpy()
            answer_latents_np = answer_latents.cpu().numpy()

            # collect answers for every batch
            for i in range(len(pred_np)):
                qid = dataset.qids[qid_idx]
                qid_idx += 1

                latent_np = answer_latents_np[i]
                latent_results.append(latent_np)
                np.save(
                    os.path.join(self.__C.ANSWER_LATENTS_DIR, f'{qid}.npy'),
                    latent_np
                )
        print()
        
        return latent_results

    def run(self):
        # Set ckpts and log path
        ## where checkpoints will be saved
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        ## where answer latents will be saved
        Path(self.__C.ANSWER_LATENTS_DIR).mkdir(parents=True, exist_ok=True)

        # build dataset entities        
        common_data = CommonData(self.__C)
        train_set = DataSet(
            self.__C,
            common_data,
            self.__C.TRAIN_SPLITS
        )
        test_set = DataSet(
            self.__C,
            common_data,
            self.__C.EVAL_SPLITS
        )

        # forward VQA model
        train_latent_results = self.eval(train_set)
        test_latent_results = self.eval(test_set)

        # search similar examples
        train_features = np.vstack(train_latent_results)
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)

        test_features = np.vstack(test_latent_results)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

        # compute top-E similar examples for each testing input
        E = self.__C.EXAMPLE_NUM
        similar_qids = {}
        print(f'\ncompute top-{E} similar examples for each testing input')
        for i, test_qid in enumerate(tqdm(test_set.qids)):
            # cosine similarity
            dists = np.dot(test_features[i], train_features.T)
            top_E = np.argsort(-dists)[:E]
            similar_qids[test_qid] = [train_set.qids[j] for j in top_E]
        
        # save similar qids
        with open(self.__C.EXAMPLE_FILE_PATH, 'w') as f:
            json.dump(similar_qids, f)

def heuristics_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for heuristics', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)

def _get_large_config(
        img_size=480, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, max_position_embeddings=199, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    heuristics_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
