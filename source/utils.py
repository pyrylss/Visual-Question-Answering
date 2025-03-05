
import numpy as np 
import pandas as pd
from torch.nn.functional import cosine_similarity

def sort_captions_based_on_similarity(captions,raw_image,model,processor, device = "cuda", ascending = False):
  """
  Rank the qr captions based on their similarity with the image
  :param captions: The captions that will be ranked 
  :param raw_image: The PIL image object 
  :param model: The image-to-text similarity model (BLIP)
  :param processor: The image and text processor 
  :param device: Cpu or Gpu
  :param ascending: Bool variable for ranking the captions at ascending order or not 
  :returns results_df: Captions ranked 
  :returns cosine_scores: The cosine score of each caption with the image
  """
  #encode the captions
  text_input = processor(text = captions, return_tensors="pt", padding = True).to(device)
  text_embeds = model.text_encoder(**text_input)
  text_embeds = text_embeds[0]
  text_features = model.text_proj(text_embeds[:, 0, :])

  #encode the image 
  image_input = processor(images=raw_image, return_tensors="pt").to(device)
  vision_outputs = model.vision_model(**image_input)
  image_embeds = vision_outputs[0]
  image_feat = model.vision_proj(image_embeds[:, 0, :])
  
  #compute cos sim
  cosine_scores = cosine_similarity(text_features, image_feat).tolist()

  #sort captions based on the cosine scores
  captions = [x for _, x in sorted(zip(cosine_scores, captions), reverse = True)]
  cosine_scores.sort(reverse = True)
  return captions, cosine_scores
