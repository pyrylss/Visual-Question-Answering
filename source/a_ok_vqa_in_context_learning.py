import os
import torch
import pandas as pd 
import numpy as np 
from ast import literal_eval
import numpy as np
from PIL import Image
from tqdm import tqdm
from source.utils import sort_captions_based_on_similarity

import json

def val_in_context_learning_a_ok_vqa_with_beit(llama_model, 
                                               llama_tokenizer,
                                               blip_model,
                                               blip_processor,
                                               train_annotations_df, 
                                               val_annotations_df,
                                               train_captions, 
                                               val_captions, 
                                               context_examples_df, 
                                               train_images_dir,
                                               val_images_dir, 
                                               n_shots=10, 
                                               k_ensemble=5,
                                               MAX_CAPTION_LEN=30, 
                                               NO_OF_CAPTIONS_AS_CONTEXT=10, 
                                               path_to_save_preds = None,
                                               device="cpu"):

  """
  Performs n-shot in context learning using the mcan-based shot selection
  :param llama_model: The llama huggingface model
  :param llama_tokenizer: The llama huggingface tokenizer
  :param blip_model: The blip huggingface model
  :param blip_processor: The blip huggingface processor
  :param train_annotations_df: Dataframe containing the ok_vqa train annotations
  :param val_annotations_df: Dataframe containing the ok_vqa val annotations
  :param context_examples_df: Dataframe containing the mcan examples
  :param train_captions: Dataframe containing the train question-informative captions
  :param val_captions: Dataframe containing the val question-informative captions
  :param train_images_dir: The path of the folder containing the training images
  :param val_images_dir: The path of the folder containing the val images
  :param n_shots: The number of shots for the in-context few-shot learning
  :param k_ensemble: The number of ensembles
  :param MAX_CAPTION_LEN: The number of maximum words to keep for each caption
  :param NO_OF_CAPTIONS_AS_CONTEXT: The number of captions to use as context for each shot
  :param path_to_save_preds: Path to save the predictions as a csv file
  :param device: Cpu or gpu device
  :returns llama_preds_df: Dataframe containing the final predictions
  """

  llama_answers = []
  question_id_list, image_id_list = [],[]
  used_normal_caption=0

  with open("../Visual-Question-Answering/knowledge_guided_captions/aokvqa_train_captions_w_knowledge.json", 'r') as f:
        train_captions_w_knowledge = json.load(f)
  with open("../Visual-Question-Answering/knowledge_guided_captions/aokvqa_val_captions_w_knowledge.json", 'r') as f:
        val_captions_w_knowledge = json.load(f)

  train_captions_knowledge_dict = {entry["question_id"]: entry["captions"] for entry in train_captions_w_knowledge}
  val_captions_knowledge_dict = {entry["question_id"]: entry["captions"] for entry in val_captions_w_knowledge}

  for i in tqdm(range(val_annotations_df.shape[0])):
   
    test_sample = val_annotations_df.iloc[i]

    #find the context examples 
    test_sample_examples_df = context_examples_df[context_examples_df.question_id == test_sample.question_id].iloc[0]

    #perform few shot in context learning for this test sample
    pred_answer_list, pred_prob_list = [], []   
    for k in range(k_ensemble): # we use k promts for each test sample
      prompt = 'Please answer the question according to the context.\n===\n'
      n_shots_var = n_shots
      for ni in range(n_shots_var):
        #take the id of the n-th shot
        context_key = test_sample_examples_df['similar_examples'][ni+n_shots_var*k]
        context_key = train_annotations_df[train_annotations_df.question_id == context_key] 
        if context_key.shape[0]!=0: #if this exist in training samples
          context_key = context_key.iloc[0]
        else:
          n_shots_var+=1
          continue
        raw_image = Image.open(train_images_dir+context_key.image_path)

        #get captions
        context_key_captions = train_captions[train_captions.question_id==context_key.question_id].iloc[0].captions

        context_key_captions_w_knowledge = train_captions_knowledge_dict.get(context_key.question_id, [])

        context_key_captions_w_knowledge = [c["caption"] for c in context_key_captions_w_knowledge]
    
        context_key_captions_w_knowledge, _ = sort_captions_based_on_similarity(context_key_captions_w_knowledge,raw_image=raw_image,model=blip_model,processor=blip_processor,device = device, ascending = False)

        context_key_captions, _ = sort_captions_based_on_similarity(context_key_captions,raw_image=raw_image,model=blip_model,processor=blip_processor,device = device, ascending = False)
        context_key_answers = train_annotations_df[train_annotations_df.question_id==context_key.question_id].iloc[0].direct_answers
        most_common_answer = max(set(context_key_answers), key=context_key_answers.count) #most common answer for this context example

        prompt += 'Context:\nStart of Context:\n' 
        for j,caption in enumerate(context_key_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
          caption = " ".join(caption.split()[:MAX_CAPTION_LEN]) #truncate
          if j < NO_OF_CAPTIONS_AS_CONTEXT-1:
            prompt += '%s,\n'%caption
          else:
            prompt += '%s\n%s\nEnd of Context\n'%(caption, context_key_captions_w_knowledge[0])
        prompt += 'Question: %s\nAnswer: %s\n\n===\n'%(context_key.question,most_common_answer)

      #get captions of the test sample
      test_sample_captions = val_captions[val_captions.question_id==test_sample.question_id].iloc[0].captions
      raw_test_image = Image.open(val_images_dir+test_sample.image_path)

      test_sample_captions_w_knowledge = val_captions_knowledge_dict.get(test_sample.question_id, [])
      test_sample_captions_w_knowledge = [c["caption"] for c in test_sample_captions_w_knowledge]
      
      test_sample_captions_w_knowledge, _ = sort_captions_based_on_similarity(test_sample_captions_w_knowledge,raw_image=raw_test_image,model=blip_model,processor=blip_processor,device = device, ascending = False)

      #sort the captions based on the cos sim 
      test_sample_captions, _ = sort_captions_based_on_similarity(test_sample_captions,raw_image=raw_test_image,model=blip_model,processor=blip_processor,device = device, ascending = False)

      prompt += 'Context:\nStart of Context:\n' 
      for j,caption in enumerate(test_sample_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
        caption = " ".join(caption.split()[:MAX_CAPTION_LEN]) #truncatte
        if j < NO_OF_CAPTIONS_AS_CONTEXT-1:
          prompt += '%s,\n'%caption
        else:
          prompt += '%s\n%s\nEnd of Context\n'%(caption, test_sample_captions_w_knowledge[0])
          
      prompt += 'Question: %s\nAnswer:'%test_sample.question

      inputs = llama_tokenizer(prompt, return_tensors="pt") 
      prompt_tokens = inputs.input_ids.shape[1] # to ignore the question

      # Generate
      input_ids = inputs.input_ids.to(device)
      outputs = llama_model.generate(input_ids, max_length=prompt_tokens + 5, num_beams=2, return_dict_in_generate=True, output_scores=True, num_return_sequences=1,
                                     do_sample=False, temperature = 1.0, top_p = 1.0)
      
      outputs_sequences, outputs_sequences_scores = outputs.sequences, outputs.sequences_scores
      pred_answer = llama_tokenizer.batch_decode(outputs_sequences[:,prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

      pred_answer_list.append(pred_answer)
      pred_prob_list.append(np.exp(outputs_sequences_scores.item()))

    #take the sequence with the max score 
    max_prob = max(pred_prob_list)
    max_index = pred_prob_list.index(max_prob)
    llama_answers.append(pred_answer_list[max_index])

    print(f"Prompt for {test_sample.question_id}\n")
    print(prompt)
    print(f"Predicted answer: {pred_answer_list[max_index]}")
    
    question_id_list.append(test_sample.question_id)
    image_id_list.append(test_sample.image_id)
    if i%10==0 and i>0: #save preds every 100 samples
      llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':llama_answers})
      llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
      llama_preds_df.to_csv(path_to_save_preds,index=False)
  #save the predictions
  llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':llama_answers})
  llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
  llama_preds_df.to_csv(path_to_save_preds,index=False)

  print(f"Used normal PC caption {used_normal_caption} times")
  return llama_preds_df


def test_in_context_learning_a_ok_vqa_with_beit(llama_model,
                                                llama_tokenizer,
                                                blip_model,
                                                blip_processor,
                                                train_annotations_df, 
                                                test_annotations_df,
                                                train_captions, 
                                                test_captions, 
                                                context_examples_df, 
                                                train_images_dir,
                                                test_images_dir, 
                                                n_shots=10,
                                                k_ensemble=5, 
                                                MAX_CAPTION_LEN=30,
                                                NO_OF_CAPTIONS_AS_CONTEXT=9, 
                                                path_to_save_preds = None,
                                                device="cpu"):
  
  """
  Performs n-shot in context learning using the mcan-based shot selection
  :param llama_model: The llama huggingface model
  :param llama_tokenizer: The llama huggingface tokenizer
  :param blip_model: The blip huggingface model
  :param blip_processor: The blip huggingface processor
  :param train_annotations_df: Dataframe containing the ok_vqa train annotations
  :param val_annotations_df: Dataframe containing the ok_vqa val annotations
  :param context_examples_df: Dataframe containing the mcan examples
  :param train_captions: Dataframe containing the train question-informative captions
  :param val_captions: Dataframe containing the val question-informative captions
  :param train_images_dir: The path of the folder containing the training images
  :param val_images_dir: The path of the folder containing the val images
  :param n_shots: The number of shots for the in-context few-shot learning
  :param k_ensemble: The number of ensembles
  :param MAX_CAPTION_LEN: The number of maximum words to keep for each caption
  :param NO_OF_CAPTIONS_AS_CONTEXT: The number of captions to use as context for each shot
  :param path_to_save_preds: Path to save the predictions as a csv file
  :param device: Cpu or gpu device
  :returns llama_preds_df: Dataframe containing the final predictions
  """

  llama_answers = []
  question_id_list, image_id_list = [],[]
  used_normal_caption=0

  train_captions_knowledge_dict = {}

  with open("../Visual-Question-Answering/knowledge_guided_captions/aokvqa_train_captions_w_knowledge.json", 'r') as f:
        train_captions_w_knowledge = json.load(f)
        train_captions_knowledge_dict.update({entry["question_id"]: entry["captions"] for entry in train_captions_w_knowledge})

  with open("../Visual-Question-Answering/knowledge_guided_captions/aokvqa_val_captions_w_knowledge.json", 'r') as f:
        val_captions_w_knowledge = json.load(f)
        train_captions_knowledge_dict.update({entry["question_id"]: entry["captions"] for entry in val_captions_w_knowledge})

  with open("../Visual-Question-Answering/knowledge_guided_captions/aokvqa_test_captions_w_knowledge.json", 'r') as f:
        test_captions_w_knowledge = json.load(f)

  test_captions_knowledge_dict = {entry["question_id"]: entry["captions"] for entry in test_captions_w_knowledge}

  for i in tqdm(range(test_annotations_df.shape[0])):
   
    test_sample = test_annotations_df.iloc[i]

    #find the context examples 
    test_sample_examples_df = context_examples_df[context_examples_df.question_id == test_sample.question_id].iloc[0]

    #perform few shot in context learning for this test sample
    pred_answer_list, pred_prob_list = [], []   
    for k in range(k_ensemble): # we use k promts for each test sample
      prompt = 'Please answer the question according to the context.\n===\n'
      n_shots_var = n_shots
      for ni in range(n_shots_var):
        #take the id of the n-th shot
        context_key = test_sample_examples_df['similar_examples'][ni+n_shots_var*k]
        context_key = train_annotations_df[train_annotations_df.question_id == context_key] 
        if context_key.shape[0]!=0: #if this exist in training samples
          context_key = context_key.iloc[0]
        else:
          n_shots_var+=1
          continue
        raw_image = Image.open(train_images_dir+context_key.image_path)

        context_key_captions = train_captions[train_captions.question_id==context_key.question_id].iloc[0].captions

        context_key_captions_w_knowledge = train_captions_knowledge_dict.get(context_key.question_id, [])
      
        context_key_captions_w_knowledge = [c["caption"] for c in context_key_captions_w_knowledge]
       
        context_key_captions_w_knowledge, cos_scores = sort_captions_based_on_similarity(context_key_captions_w_knowledge,raw_image=raw_image,model=blip_model,processor=blip_processor,device = device, ascending = False)
       
        #sort the captions based on the cos sim 
        context_key_captions, cos_scores = sort_captions_based_on_similarity(context_key_captions,raw_image=raw_image,model=blip_model,processor=blip_processor,device = device,ascending = False)
        context_key_answers = train_annotations_df[train_annotations_df.question_id==context_key.question_id].iloc[0].direct_answers
        
        if isinstance(context_key_answers, str):
            context_key_answers = literal_eval(context_key_answers)

        most_common_answer = max(set(context_key_answers), key=context_key_answers.count) #most common answer for this context example

        prompt += 'Context:\nStart of Context:\n' 
        for j,caption in enumerate(context_key_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
          caption = " ".join(caption.split()[:MAX_CAPTION_LEN]) #truncate
          if j < NO_OF_CAPTIONS_AS_CONTEXT-1:
            prompt += '%s,\n'%caption
          else:
            prompt += '%s\n%s\nEnd of Context\n'%(caption, context_key_captions_w_knowledge[0])

        prompt += 'Question: %s\nAnswer: %s\n\n===\n'%(context_key.question,most_common_answer)

      test_sample_captions = test_captions[test_captions.question_id==test_sample.question_id].iloc[0].captions
      raw_test_image = Image.open(test_images_dir+test_sample.image_path)

      test_sample_captions_w_knowledge = test_captions_knowledge_dict.get(test_sample.question_id, [])
      
      test_sample_captions_w_knowledge = [c["caption"] for c in test_sample_captions_w_knowledge]
      
      test_sample_captions_w_knowledge, cos_scores = sort_captions_based_on_similarity(test_sample_captions_w_knowledge,raw_image=raw_test_image,model=blip_model,processor=blip_processor,device = device, ascending = False)
      #sort the captions based on the cos sim 
      test_sample_captions, cos_scores = sort_captions_based_on_similarity(test_sample_captions,raw_image=raw_test_image,model=blip_model,processor=blip_processor,device = device, ascending = False)

      prompt += 'Context:\nStart of Context:\n' 
      for j,caption in enumerate(test_sample_captions[:NO_OF_CAPTIONS_AS_CONTEXT]):
        caption = " ".join(caption.split()[:MAX_CAPTION_LEN]) #truncatte
        if j < NO_OF_CAPTIONS_AS_CONTEXT-1:
          prompt += '%s,\n'%caption
        else:
          prompt += '%s\n%s\nEnd of Context\n'%(caption, test_sample_captions_w_knowledge[0])
      prompt += 'Question: %s\nAnswer:'%test_sample.question

      inputs = llama_tokenizer(prompt, return_tensors="pt") 
      prompt_tokens = inputs.input_ids.shape[1] # to ignore the question

      # Generate
      input_ids = inputs.input_ids.to(device)
      outputs = llama_model.generate(input_ids, max_length=prompt_tokens + 5, num_beams=2, return_dict_in_generate=True, output_scores=True, num_return_sequences=1,
                                     do_sample=False, temperature = 1.0, top_p = 1.0)
      
      outputs_sequences, outputs_sequences_scores = outputs.sequences, outputs.sequences_scores
      pred_answer = llama_tokenizer.batch_decode(outputs_sequences[:,prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

      pred_answer_list.append(pred_answer)
      pred_prob_list.append(np.exp(outputs_sequences_scores.item()))

    #take the sequence with the max score 
    max_prob = max(pred_prob_list)
    max_index = pred_prob_list.index(max_prob)
    llama_answers.append(pred_answer_list[max_index])

    print(f"Prompt for {test_sample.question_id}\n")
    print(prompt)
    print(f"Predicted answer: {pred_answer_list[max_index]}")
    
    question_id_list.append(test_sample.question_id)
    image_id_list.append(test_sample.image_id)
    if i%10==0 and i>0: #save preds every 100 samples
      llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':llama_answers})
      llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
      llama_preds_df.to_csv(path_to_save_preds, index=False)

  #save the predictions
  llama_preds_df = pd.DataFrame({'question_id':question_id_list,'image_id':image_id_list,'llama_answer':llama_answers})
  llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=","").strip())
  llama_preds_df.to_csv(path_to_save_preds, index=False)

  print(f"Used normal PC caption {used_normal_caption} times")

  return llama_preds_df
