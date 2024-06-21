import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import ast
import time
import csv
import json
import os


# you may change it to any model you need to experiment with, such as "meta-llama/Meta-Llama-3-8B-Instruct" or "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", cache_dir=" ")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    device_map="auto",
    cache_dir=" "
)

df = pd.read_csv("<your_file>.csv")


output_dir = "<data>_results"
os.makedirs(output_dir, exist_ok=True)

# List of context perturbations including raw context
perturbations = ['Question', 'Question_red_herrings', 'Question_typos', 'Question_HP', 'Question_Leet', 'Question_It_cleft', 'Question_Wh_cleft', 'Question_stress', 'Question_checklist', 'Question_comp']

# Process each perturbation for Gemma-2B/7B
for perturbation in perturbations:
    prediction_reasonings = []
    
    for index, row in df.iterrows():
        system_prompt = "You are a knowledge expert specializing in complex reasoning, and your task is to answer multiple-choice questions accurately and efficiently."
        question_perturbed = row[perturbation]
        answerA = row['Option A']
        answerB = row['Option B']
        answerC = row['Option C']
        answerD = row['Option D']

        input_text = (
            # 5 sample demonstration here
            f"Question: \n\n"
            f"A: \n"
            f"B: \n"
            f"C: \n"
            f"D: \n\n"
            f"OUTPUT: REASONING:"
            # real question
            f"Question: {question_perturbed}\n\n"
            f"A: {answerA}\n"
            f"B: {answerB}\n"
            f"C: {answerC}\n"
            f"D  {answerD}\n\n"
        )
        messages = [{"role": "user", "content": system_prompt + '\n\n' + input_text},]
        
        prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
        inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1024)


        text = tokenizer.decode(outputs[0])


        response = text.split('<start_of_turn>model')
        prediction_reasonings.append(response[1])

    
    # Add the responses to the DataFrame
    df[f'Gemma_7B_{perturbation}'] = prediction_reasonings
    
    # Save the results to a CSV file
    output_file = os.path.join(output_dir, f"Gemma_7B_{perturbation}.csv")
    df[[perturbation, 'Option A', 'Option B', 'Option C', 'Option D', 'Answer', f'Gemma_7B_{perturbation}']].to_csv(output_file, index=False)

    print(f"Saved results for {perturbation} to {output_file}")
    

    
# Process each perturbation for Llama3-8B/70B
for perturbation in perturbations:
    prediction_reasonings = []
    
    for index, row in df.iterrows():
        
        system_message = {
        "role": "system",
        "content": "You are a knowledge expert specializing in complex reasoning, and your task is to answer multiple-choice questions accurately and efficiently."
    }
        question_perturbed = row[perturbation]
        answerA = row['Option A']
        answerB = row['Option B']
        answerC = row['Option C']
        answerD = row['Option D']
        
        input_text = (
            # 5 sample demonstration here
            f"Question: \n\n"
            f"A: \n"
            f"B: \n"
            f"C: \n"
            f"D: \n\n"
            f"OUTPUT: REASONING:"
            # real question
            f"Question: {question_perturbed}\n\n"
            f"A: {answerA}\n"
            f"B: {answerB}\n"
            f"C: {answerC}\n"
            f"D  {answerD}\n\n"
        )
        
        user_message = {
                "role": "user",
                "content": input_text
            }

        messages = [system_message, user_message]

        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0,
        )
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
    
        prediction_reasonings.append(decoded_response)

    
    # Add the responses to the DataFrame
    df[f'Llama3_8B_{perturbation}'] = prediction_reasonings
    
    # Save the results to a CSV file
    output_file = os.path.join(output_dir, f"Llama3_8B_{perturbation}.csv")
    df[[perturbation, 'Option A', 'Option B', 'Option C', 'Option D', 'Answer', f'Llama3_8B_{perturbation}']].to_csv(output_file, index=False)

    print(f"Saved results for {perturbation} to {output_file}")
    
    
    
# Process each perturbation for Phi-3-Mini/Medium
for perturbation in perturbations:
    prediction_reasonings = []
    
    for index, row in df.iterrows():
        system_prompt = "You are a knowledge expert specializing in complex reasoning, and your task is to answer multiple-choice questions accurately and efficiently."
        
        question_perturbed = row[perturbation]
        answerA = row['Option A']
        answerB = row['Option B']
        answerC = row['Option C']
        answerD = row['Option D']
        
        input_text = (
            # 5 sample demonstration here
            f"Question: \n\n"
            f"A: \n"
            f"B: \n"
            f"C: \n"
            f"D: \n\n"
            f"OUTPUT: REASONING:"
            # real question
            f"Question: {question_perturbed}\n\n"
            f"A: {answerA}\n"
            f"B: {answerB}\n"
            f"C: {answerC}\n"
            f"D  {answerD}\n\n"
        )
        
        messages = [
       {"role": "user", "content": system_prompt + '\n\n' + input_text},
      ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0,
        )


        text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        prediction_reasonings.append(text)

    
    # Add the responses to the DataFrame
    df[f'Phi3_{perturbation}'] = prediction_reasonings
    
    # Save the results to a CSV file
    output_file = os.path.join(output_dir, f"Phi3_{perturbation}.csv")
    df[[perturbation, 'Option A', 'Option B', 'Option C', 'Option D', 'Answer', f'Phi3_{perturbation}']].to_csv(output_file, index=False)

    print(f"Saved results for {perturbation} to {output_file}")
