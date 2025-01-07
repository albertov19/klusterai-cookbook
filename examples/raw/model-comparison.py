import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


import urllib.request
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import json
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000, 'display.max_colwidth', 500)

# Set up the client
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

def create_tasks(df, task_type, system_prompt, model):
    tasks = []
    for index, row in df.iterrows():
        content = row['Overview']
        
        task = {
            "custom_id": f"{task_type}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
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
df = pd.read_csv('imdb_top_1000.csv', usecols=['Series_Title', 'Overview', 'Genre'])
df.head(3)

SYSTEM_PROMPT = '''
    You are a helpful assitant that classifies movie genres based on the movie description. Choose one of the following options: 
    Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western.
    Provide your response as a single word with the matching genre. Don't include punctuation.
    '''

# Define models
models = {
        '8B':"klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
        '70B':"klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
        '405B':"klusterai/Meta-Llama-3.1-405B-Instruct-Turbo",
        }

# Process each model: create tasks, run jobs, and get results
for name, model in models.items():
    task_list = create_tasks(df, task_type='assistant', system_prompt=SYSTEM_PROMPT, model=model)
    filename = save_tasks(task_list, task_type='assistant')
    job = create_batch_job(filename)
    monitor_job_status(client=client, job_id=job.id, task_type=f'{name} model')
    df[f'{name}_genre'] = get_results(client=client, job_id=job.id)

# Calculate accuracy for each model
accuracies = {}
for name, _ in models.items():
    accuracy = df.apply(lambda row: row[f'{name}_genre'] in row['Genre'].split(', '), axis=1).mean()
    accuracies[name] = accuracy

# Create the bar plot
fig, ax = plt.subplots()
bars = ax.bar(accuracies.keys(), accuracies.values(), edgecolor='black')
ax.bar_label(bars, label_type='center', color='white', fmt="%.3f")
ax.set_ylim(0, max(accuracies.values())+ 0.01)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')

ax.set_title('Classification accuracy by model')
plt.show()


