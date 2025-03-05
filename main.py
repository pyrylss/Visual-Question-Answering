import torch
import numpy as np
import pandas as pd
from ast import literal_eval
import os, sys
import json
import random
import yaml


from source import evaluation, eval_MC
from source.ok_vqa_in_context_learning import val_in_context_learning_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning import  val_in_context_learning_a_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning import test_in_context_learning_a_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning_MC import val_in_context_learning_a_ok_vqa_with_beit_MC, test_in_context_learning_a_ok_vqa_with_beit_MC

from configs.task_cfgs import Cfgs
from config import get_config
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoProcessor, BlipForImageTextRetrieval



#get confing variables 
cnf = get_config(sys.argv)

if cnf.finetune=="True":
    from beit_finetuning import get_runner

    __C = Cfgs(cnf)
    with open(cnf.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = get_runner(__C)
    runner.run()

if cnf.gen_examples=="True":
    from beit_finetuning import get_runner

    __C = Cfgs(cnf)
    with open(cnf.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = get_runner(__C)
    runner.run()


#set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#unfolding common params
train_images_dir = cnf.train_images_dir
val_images_dir = cnf.val_images_dir
test_images_dir = cnf.test_images_dir

n_shots = cnf.n_shots
k_ensemble = cnf.k_ensemble
no_of_captions = cnf.no_of_captions
path_to_save_preds = cnf.path_to_save_preds 

#load Llama model
# "meta-llama/Llama-2-13b-hf"
llama_model = LlamaForCausalLM.from_pretrained(cnf.llama_path)
llama_tokenizer = LlamaTokenizer.from_pretrained(cnf.llama_path)
llama_model = llama_model.to(device, dtype=torch.float16)

#load the blip model 
blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip_model = blip_model.to(device)

#load annotations
train_annotations_df = pd.read_csv(cnf.train_annotations_path)

if cnf.TASK == "ok" or cnf.TASK == "aok_val":
    val_annotations_df = pd.read_csv(cnf.val_annotations_path)
else:
    test_annotations_df = pd.read_csv(cnf.test_annotations_path)
 
with open(cnf.examples_path, "rb") as input:
    examples = json.load(input)
beit_examples_df = pd.DataFrame({'question_id' : examples.keys(), 'similar_examples' : examples.values()})

#load captions 
train_captions = pd.read_csv(cnf.train_captions_path)
train_captions.captions = train_captions.captions.apply(literal_eval)

if cnf.TASK == "ok" or cnf.TASK == "aok_val":
    val_captions = pd.read_csv(cnf.val_captions_path)
    val_captions.captions = val_captions.captions.apply(literal_eval)
else:
    test_captions = pd.read_csv(cnf.test_captions_path)
    test_captions.captions = test_captions.captions.apply(literal_eval)


if __name__ == "__main__":
    if cnf.TASK == "ok":
        #apply literal eval to the answers
        train_annotations_df.answers = train_annotations_df.answers.apply(literal_eval)
        val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)

        beit_examples_df['question_id'] = beit_examples_df['question_id'].astype('int')
        results_df = val_in_context_learning_ok_vqa_with_beit(llama_model=llama_model, 
                                                                llama_tokenizer=llama_tokenizer,
                                                                blip_model=blip_model,
                                                                blip_processor=blip_processor,
                                                                train_annotations_df=train_annotations_df,
                                                                val_annotations_df=val_annotations_df,
                                                                train_captions=train_captions, 
                                                                val_captions=val_captions,
                                                                context_examples_df=beit_examples_df, 
                                                                train_images_dir=train_images_dir, 
                                                                val_images_dir=val_images_dir,
                                                                n_shots=n_shots, 
                                                                k_ensemble=k_ensemble,
                                                                MAX_CAPTION_LEN=30,
                                                                NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                path_to_save_preds=path_to_save_preds,
                                                                device=device)
    
    elif cnf.TASK == "aok_val":
        #apply literal eval to the answers
        train_annotations_df.direct_answers = train_annotations_df.direct_answers.apply(literal_eval)

        #apply literal eval to the answers
        val_annotations_df.direct_answers = val_annotations_df.direct_answers.apply(literal_eval)
        if cnf.multiple_choice=="True":
            results_df = val_in_context_learning_a_ok_vqa_with_beit_MC(llama_model=llama_model,
                                                                llama_tokenizer=llama_tokenizer,
                                                                blip_model=blip_model,
                                                                blip_processor=blip_processor,
                                                                train_annotations_df=train_annotations_df,
                                                                val_annotations_df=val_annotations_df, 
                                                                train_captions=train_captions, 
                                                                val_captions=val_captions,
                                                                context_examples_df=beit_examples_df, 
                                                                train_images_dir=train_images_dir, 
                                                                val_images_dir=val_images_dir,
                                                                n_shots=n_shots, 
                                                                k_ensemble=k_ensemble, 
                                                                MAX_CAPTION_LEN=30, 
                                                                NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                path_to_save_preds=path_to_save_preds,
                                                                device=device)
        else:
            results_df = val_in_context_learning_a_ok_vqa_with_beit(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    val_annotations_df=val_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    val_captions=val_captions,
                                                                    context_examples_df=beit_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    val_images_dir=val_images_dir,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
    elif cnf.TASK == "aok_test":
        if cnf.multiple_choice=="True":
            results_df = test_in_context_learning_a_ok_vqa_with_beit_MC(llama_model=llama_model,
                                                                llama_tokenizer=llama_tokenizer,
                                                                blip_model=blip_model,
                                                                blip_processor=blip_processor,
                                                                train_annotations_df=train_annotations_df,
                                                                test_annotations_df=test_annotations_df, 
                                                                train_captions=train_captions, 
                                                                test_captions=test_captions,
                                                                context_examples_df=beit_examples_df, 
                                                                train_images_dir=train_images_dir, 
                                                                test_images_dir=test_images_dir,
                                                                n_shots=n_shots, 
                                                                k_ensemble=k_ensemble, 
                                                                MAX_CAPTION_LEN=30, 
                                                                NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                path_to_save_preds=path_to_save_preds,
                                                                device=device)
        else:
            results_df = test_in_context_learning_a_ok_vqa_with_beit(llama_model=llama_model,
                                                                llama_tokenizer=llama_tokenizer,
                                                                blip_model=blip_model,
                                                                blip_processor=blip_processor,
                                                                train_annotations_df=train_annotations_df,
                                                                test_annotations_df=test_annotations_df, 
                                                                train_captions=train_captions, 
                                                                test_captions=test_captions,
                                                                context_examples_df=beit_examples_df,
                                                                train_images_dir=train_images_dir, 
                                                                test_images_dir=test_images_dir,
                                                                n_shots=n_shots, 
                                                                k_ensemble=k_ensemble, 
                                                                MAX_CAPTION_LEN=30, 
                                                                NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                path_to_save_preds=path_to_save_preds,
                                                                device=device)
                                        
    #evaluate the predictions (only for val sets)
    if cnf.TASK == "ok" or cnf.TASK == "aok_val":
        results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
        if cnf.TASK == "ok":
            results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers']), axis = 1)
            print("\n========")
            print("VQA acc: ", np.round(results_df.acc.mean(),4))
            print("=========")
        else:
            if cnf.multiple_choice=='True':
                eval_MC.eval_MC(results_df)
            else:
                results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['direct_answers']), axis = 1)
                print("\n========")
                print("VQA acc: ", np.round(results_df.acc.mean(),4))
                print("==========")
    


    





                   


  