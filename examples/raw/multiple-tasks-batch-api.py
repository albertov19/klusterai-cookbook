import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


import os
import urllib.request
import pandas as pd
import requests
from openai import OpenAI
import time
import json
from IPython.display import clear_output, display

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000, 'display.max_colwidth', 500)

# Set up the client
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

# Choose your dataset:
# AMZ musical instrument reviews dataset:
url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz"

# IMDB Top 1000 sample dataset:
#url = "https://raw.githubusercontent.com/kluster-ai/klusterai-cookbook/refs/heads/main/data/imdb_top_1000.csv"

# AG News sample dataset:
#url = "https://raw.githubusercontent.com/kluster-ai/klusterai-cookbook/refs/heads/main/data/ag_news.csv"

def fetch_dataset(url, file_path=None):
    # Set the default file path based on the URL if none is provided
    if not file_path:
        file_path = os.path.join("data", os.path.basename(url))

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download the file
    urllib.request.urlretrieve(url, file_path)
    print(f"Dataset downloaded and saved as {file_path}")

    # Load and process the dataset based on URL content
    if "imdb_top_1000.csv" in url:
        df = pd.read_csv(file_path)
        df['text'] = df['Series_Title'].astype(str) + ": " + df['Overview'].astype(str)
        df = df[['text']]
    elif "ag_news" in url:
        df = pd.read_csv(file_path, header=None, names=["label", "title", "description"])
        df['text'] = df['title'].astype(str) + ": " + df['description'].astype(str)
        df = df[['text']]
    elif "reviews_Musical_Instruments_5.json.gz" in url:
        df = pd.read_json(file_path, compression='gzip', lines=True)
        df.rename({'reviewText': 'text'}, axis=1, inplace=True)
        df = df[['text']]
    else:
        raise ValueError("URL does not match any known dataset format.")

    return df.tail(5).reset_index(drop=True)

df = fetch_dataset(url=url, file_path=None)
df.head()

SYSTEM_PROMPTS = {
    'sentiment': '''
    Analyze the sentiment of the given text. Provide only a JSON object with the following structure:
    {
        "sentiment": string, // "positive", "negative", or "neutral"
        "confidence": float, // A value between 0 and 1 indicating your confidence in the sentiment analysis
    }
    ''',

    'translation': '''
    Translate the given text from English to Spanish, paraphrase, rewrite or perform cultural adaptations for the text to make sense in Spanish. Provide only a JSON object with the following structure:
    {
        "translation": string, // The Spanish translation
        "notes": string // Any notes about the translation, such as cultural adaptations or challenging phrases (max 500 words). Write this mainly in english.
    }
    ''',

    'summary': '''
    Summarize the main points of the given text. Provide only a JSON object with the following structure:
    {
        "summary": string, // A concise summary of the text (max 100 words)
    }
    ''',

    'topic_classification': '''
    Classify the main topic of the given text based on the following categories: "politics", "sports", "technology", "science", "business", "entertainment", "health", "other". Provide only a JSON object with the following structure:
    {
        "category": string, // The primary category of the provided text
        "confidence": float, // A value between 0 and 1 indicating confidence in the classification
    }
    ''',

    'keyword_extraction': '''
    Extract relevant keywords from the given text. Provide only a JSON object with the following structure:
    {
        "keywords": string[], // An array of up to 5 keywords that best represent the text content
        "context": string // Briefly explain how each keyword is relevant to the text (max 200 words)
    }
    '''
}

def create_inference_file(df, inference_type, system_prompt):
    inference_list = []
    for index, row in df.iterrows():
        content = row['text']

        request = {
            "custom_id": f"{inference_type}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "klusterai/Meta-Llama-3.1-405B-Instruct-Turbo",
                "temperature": 0.5,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
            }
        }
        inference_list.append(request)
    return inference_list

def save_inference_file(inference_list, inference_type):
    filename = f"data/inference_request_{inference_type}.jsonl"
    with open(filename, 'w') as file:
        for request in inference_list:
            file.write(json.dumps(request) + '\n')
    return filename

inference_requests = []

for inference_type, system_prompt in SYSTEM_PROMPTS.items():
    inference_list = create_inference_file(df, inference_type, system_prompt)
    filename = save_inference_file(inference_list, inference_type)
    inference_requests.append((inference_type, filename))

def create_inference_job(file_name):
    print(f"Creating request for {file_name}")
    inference_input_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )

    inference_job = client.batches.create(
        input_file_id=inference_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return inference_job

inference_jobs = []

for inference_type, file_name in inference_requests:
    job = create_inference_job(file_name)
    inference_jobs.append((f"{inference_type}", job))

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

all_completed = False
while not all_completed:
    all_completed = True
    output_lines = []

    for i, (job_type, job) in enumerate(inference_jobs):
        updated_job = client.batches.retrieve(job.id)
        inference_jobs[i] = (job_type, updated_job)

        if updated_job.status.lower() != "completed":
            all_completed = False
            completed = updated_job.request_counts.completed
            total = updated_job.request_counts.total
            output_lines.append(f"{job_type.capitalize()} job status: {updated_job.status} - Progress: {completed}/{total}")
        else:
            output_lines.append(f"{job_type.capitalize()} job completed!")

    # Clear the output and display updated status
    clear_output(wait=True)
    for line in output_lines:
        display(line)

    if not all_completed:
        time.sleep(10)

for job_type, job in inference_jobs:
    inference_job = client.batches.retrieve(job.id)
    result_file_id = inference_job.output_file_id
    result = client.files.content(result_file_id).content
    results = parse_json_objects(result)

    for res in results:
        inference_id = res['custom_id']
        index = inference_id.split('-')[-1]
        result = res['response']['body']['choices'][0]['message']['content']
        text = df.iloc[int(index)]['text']
        print(f'\n -------------------------- \n')
        print(f"Inference ID: {inference_id}. \n\nTEXT: {text}\n\nRESULT: {result}")


