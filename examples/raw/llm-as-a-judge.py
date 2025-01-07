import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


import os
import urllib.request
import pandas as pd
import numpy as np
import random
import requests
from openai import OpenAI
import time
import json
from IPython.display import clear_output, display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000, 'display.max_colwidth', 500)

# Set up the client
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

def create_tasks(user_contents, system_prompt, task_type, model):
    tasks = []
    for index, user_content in enumerate(user_contents):
        task = {
            "custom_id": f"{task_type}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
        }
        tasks.append(task)
    return tasks

def save_tasks(tasks, task_type):
    filename = f"batch_tasks_{task_type}.jsonl"
    with open(filename, 'w') as file:
        for task in tasks:
            file.write(json.dumps(task) + '\n')
    return filename

def create_batch_job(file_name):
    print(f"Creating batch job for {file_name}")
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch_job

def parse_json_objects(data_string):
    if isinstance(data_string, bytes):
        data_string = data_string.decode('utf-8')

    json_strings = data_string.strip().split('\n')
    json_objects = []

    for json_str in json_strings:
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    return json_objects

def monitor_job_status(client, job_id, task_type):
    all_completed = False

    while not all_completed:
        all_completed = True
        output_lines = []

        updated_job = client.batches.retrieve(job_id)

        if updated_job.status.lower() != "completed":
            all_completed = False
            completed = updated_job.request_counts.completed
            total = updated_job.request_counts.total
            output_lines.append(f"{task_type.capitalize()} job status: {updated_job.status} - Progress: {completed}/{total}")
        else:
            output_lines.append(f"{task_type.capitalize()} job completed!")

        # Clear the output and display updated status
        clear_output(wait=True)
        for line in output_lines:
            display(line)

        if not all_completed:
            time.sleep(10)

def get_results(client, job_id):
    batch_job = client.batches.retrieve(job_id)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    results = parse_json_objects(result)
    answers = []
    
    for res in results:
        result = res['response']['body']['choices'][0]['message']['content']
        answers.append(result)
    
    return answers

# IMDB Top 1000 dataset:
url = "https://raw.githubusercontent.com/kluster-ai/klusterai-cookbook/refs/heads/main/data/imdb_top_1000.csv"
urllib.request.urlretrieve(url,filename='imdb_top_1000.csv')

# Load and process the dataset based on URL content
df = pd.read_csv('imdb_top_1000.csv', usecols=['Series_Title', 'Overview', 'Genre']).tail(300)
df[['Series_Title','Overview']].head(3)

prompt_dict = {
    "ASSISTANT_PROMPT" : '''
        You are a helpful assitant that classifies movie genres based on the movie description. Choose one of the following options: 
        Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western.
        Provide your response as a single word with the matching genre. Don't include punctuation.
    ''',
    "USER_CONTENTS" : df['Overview'].tolist()
}

task_list = create_tasks(user_contents=prompt_dict["USER_CONTENTS"], 
                         system_prompt=prompt_dict["ASSISTANT_PROMPT"], 
                         model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo", 
                         task_type='assistant')
filename = save_tasks(task_list, task_type='assistant')
job = create_batch_job(filename)
monitor_job_status(client=client, job_id=job.id, task_type='assistant')
df['predicted_genre'] = get_results(client=client, job_id=job.id)

prompt_dict = {
    "JUDGE_PROMPT" : '''
        You will be provided with a movie description, a list of possible genres, and a predicted movie genre made by another LLM. Your task is to evaluate whether the predicted genre is ‘correct’ or ‘incorrect’ based on the following steps and requirements.
        
        Steps to Follow:
        1. Carefully read the movie description.
        2. Determine your own classification of the genre for the movie. Do not rely on the LLM's answer since it may be incorrect. Do not rely on individual words to identify the genre; read the whole description to identify the genre.
        3. Read the LLM answer (enclosed in double quotes) and evaluate if it is the correct answer by following the Evaluation Criteria mentioned below.
        4. Provide your evaluation as 'correct' or 'incorrect'.
        
        Evaluation Criteria:
        - Ensure the LLM answer (enclosed in double quotes) is one of the provided genres. If it is not listed, the evaluation should be ‘incorrect’.
        - If the LLM answer (enclosed in double quotes) does not align with the movie description, the evaluation should be ‘incorrect’.
        - The first letter of the LLM answer (enclosed in double quotes) must be capitalized (e.g., Drama). If it has any other capitalization, the evaluation should be ‘incorrect’.
        - All other letters in the LLM answer (enclosed in double quotes) must be lowercase. Otherwise, the evaluation should be ‘incorrect’.
        - If the LLM answer consists of multiple words, the evaluation should be ‘incorrect’.
        - If the LLM answer includes punctuation, spaces, or additional characters, the evaluation should be ‘incorrect’.
        
        Output Rules:
        - Provide the evaluation with no additional text, punctuation, or explanation.
        - The output should be in lowercase.
        
        Final Answer Format:
        evaluation
        
        Example:
        correct
    ''',
    "USER_CONTENTS" : [f'''Movie Description: {row['Overview']}.
        Available Genres: Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western
        LLM answer: "{row['predicted_genre']}"
        ''' for _, row in df.iterrows()
        ]
}

task_list = create_tasks(user_contents=prompt_dict["USER_CONTENTS"], 
                         system_prompt=prompt_dict["JUDGE_PROMPT"], 
                         task_type='judge', 
                         model="klusterai/Meta-Llama-3.1-405B-Instruct-Turbo")
filename = save_tasks(task_list, task_type='judge')
job = create_batch_job(filename)
monitor_job_status(client=client, job_id=job.id, task_type='judge')
df['judge_evaluation'] = get_results(client=client, job_id=job.id)

print('LLM Judge-determined accuracy: ',df['judge_evaluation'].value_counts(normalize=True)['correct'])

print('LLM ground truth accuracy: ',df.apply(lambda row: row['predicted_genre'] in row['Genre'].split(', '), axis=1).mean())
