import os
import argparse


def get_config(sysv):
    parser = argparse.ArgumentParser(description='In-context learning variables.')
    parser.add_argument('--train_annotations_path', type=str, default=None, help='The path to the train annotations csv file')
    parser.add_argument('--val_annotations_path', type=str, default=None, help='The path to the train annotations csv file')
    parser.add_argument('--test_annotations_path', type=str, default=None, help='The path to the train annotations csv file')

    parser.add_argument('--train_images_dir', type=str, default=None, help='Path for the training images dir')
    parser.add_argument('--val_images_dir', type=str, default=None, help='Path for the val images dir')
    parser.add_argument('--test_images_dir', type=str, default=None, help='Path for the test images dir (only for A-OK-VQA)')

    parser.add_argument('--n_shots', type=int, default=10, help='Number of shots for in-context-learning')
    parser.add_argument('--k_ensemble', type=int, default=5, help='Number of ensmembles for in-context-learning')
    parser.add_argument('--no_of_captions', type=int, default=9, help='Number of question-guided captions for in-context-learning')
    parser.add_argument('--multiple_choice', type=str, default="False", choices=["True", "False"], help='True for MC task')
    parser.add_argument('--examples_path', type=str, default=None, help='The path to the json file containing the mcan examples')
    parser.add_argument('--llama_path', type=str, default=None, help='The path to the llama (1 or 2) weights')
    parser.add_argument('--blip_train_question_embedds_path', type=str, default=None, help='The path to the normalized blip train question embeddings')
    parser.add_argument('--blip_train_image_embedds_path', type=str, default=None, help='The path to the normalized blip train image embeddings')
    parser.add_argument('--blip_val_question_embedds_path', type=str, default=None, help='The path to the normalized blip val question embeddings')
    parser.add_argument('--blip_val_image_embedds_path', type=str, default=None, help='The path to the normalized blip val image embeddings')
    parser.add_argument('--blip_test_question_embedds_path', type=str, default=None, help='The path to the normalized blip test question embeddings (only for A-OK-VQA)')
    parser.add_argument('--blip_test_image_embedds_path', type=str, default=None, help='The path to the normalized blip test image embeddings (only for A-OK-VQA)')

    parser.add_argument('--train_captions_path', type=str, default=None, help='The path to the train question informative captions')
    parser.add_argument('--val_captions_path', type=str, default=None, help='The path to the val question informative captions')
    parser.add_argument('--test_captions_path', type=str, default=None, help='The path to the train question informative captions (only for A-OK-VQA)')

    parser.add_argument('--path_to_save_preds', type=str, default=None, help='Path to save the final predictions (needs to have .csv extension)')

    parser.add_argument('--finetune', type=str, default=False, help='are you running finetuning', required=True)
    parser.add_argument('--gen_examples', type=str, default=False, help='are you selecting in-context examples')
    parser.add_argument('--task', dest='TASK', help="task name, one of ['ok', 'aok_val', 'aok_test']", type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['finetune, 'heuristics']", type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str)
    parser.add_argument('--version', dest='VERSION', help='version name, output folder will be named as version name', type=str)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=99)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)

    args, _ = parser.parse_known_args(sysv)

    return args 

