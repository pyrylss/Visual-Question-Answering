python main.py \
    --task ok \
    --train_annotations_path annotations/ok_vqa/train_annots_fixed.csv.zip \
    --val_annotations_path annotations/ok_vqa/val_annots_fixed.csv.zip \
    --test_annotations_path None \
    --train_images_dir datasets/coco2014/train2014/ \
    --val_images_dir datasets/coco2014/val2014/ \
    --test_images_dir None \
    --n_shots 10 \
    --k_ensemble 5 \
    --no_of_captions 9 \
    --examples_path beit3_examples/ok_vqa/examples.json \
    --llama_path meta-llama/Llama-2-13b-hf \
    --train_captions_path question_related_captions/ok_vqa/train_data_qr_captions_csv \
    --val_captions_path question_related_captions/ok_vqa/val_data_qr_captions_csv \
    --test_captions_path None \
    --blip_train_question_embedds_path blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_train_question_embedds.csv.zip \
    --blip_train_image_embedds_path blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_train_image_embedds.csv.zip \
    --blip_val_question_embedds_path blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_val_question_embedds.csv.zip \
    --blip_val_image_embedds_path blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_val_image_embedds.csv.zip \
    --finetune False \
    --path_to_save_preds results/okvqa_val.csv
    
# python main.py \
#     --task aok_val \
#     --train_annotations_path annotations/a_ok_vqa/a_ok_vqa_train_fixed_annots.csv.zip \
#     --val_annotations_path  annotations/a_ok_vqa/a_ok_vqa_val_fixed_annots.csv.zip \
#     --test_annotations_path annotations/a_ok_vqa/a_ok_vqa_test_fixed_annots.csv.zip \
#     --train_images_dir datasets/coco2017/train2017/ \
#     --val_images_dir datasets/coco2017/val2017/ \
#     --test_images_dir datasets/coco2017/test2017/ \
#     --n_shots 10 \
#     --k_ensemble 5 \
#     --no_of_captions 9 \
#     --multiple_choice False \
#     --examples_path beit3_examples/a_ok_vqa/aokvqa_examples_val.json \
#     --llama_path meta-llama/Llama-2-13b-hf \
#     --train_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_train_qr_captions.csv.zip \
#     --val_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_val_qr_captions.csv.zip \
#     --test_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_test_qr_captions.csv.zip \
#     --blip_train_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_train_question_embedds.csv.zip \
#     --blip_train_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_train_image_embedds.csv.zip \
#     --blip_val_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_val_question_embedds.csv.zip \
#     --blip_val_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_val_image_embedds.csv.zip \
#     --blip_test_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_test_question_embedds.csv.zip \
#     --blip_test_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_test_image_embedds.csv.zip \
#     --finetune False \
#     --path_to_save_preds results/a_ok_vqa_val.csv
