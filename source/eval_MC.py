import pandas as pd
import json

def eval_MC(predictions_df):

    # Load correct answers
    with open('datasets/aokvqa/aokvqa_v1p0_val.json', 'r') as f:
        correct_answers = json.load(f)

    # Create a dictionary for quick lookup of correct answers
    correct_dict = {
        entry["question_id"]: entry["choices"][entry["correct_choice_idx"]]
        for entry in correct_answers
    }
    total_questions = 0
    correct_predictions = 0

    for _, row in predictions_df.iterrows():
        question_id = row["question_id"]
        predicted_answer = row["llama_answer"]

        # Check if the prediction matches the correct answer
        if question_id in correct_dict:
            total_questions += 1
            if predicted_answer == correct_dict[question_id]:
                correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0

    print(f"MC Accuracy: {accuracy:.2f}%")
