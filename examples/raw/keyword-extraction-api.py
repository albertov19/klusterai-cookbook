import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


from openai import OpenAI

import pandas as pd
import time
import json
from IPython.display import clear_output, display

# Set up the client
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

df = pd.DataFrame({
    "text": [
        "Chorus Frog Found Croaking in Virginia - The Southern chorus frog has been found in southeastern Virginia, far outside its previously known range. The animal had never before been reported north of Beaufort County, N.C., about 125 miles to the south.",
        "Expedition to Probe Gulf of Mexico - Scientists will use advanced technology never before deployed beneath the sea as they try to discover new creatures, behaviors and phenomena in a 10-day expedition to the Gulf of Mexico's deepest reaches.",
        "Feds Accused of Exaggerating Fire ImpactP - The Forest Service exaggerated the effect of wildfires on California spotted owls in justifying a planned increase in logging in the Sierra Nevada, according to a longtime agency expert who worked on the plan.",
        "New Method May Predict Quakes Weeks Ahead - Swedish geologists may have found a way to predict earthquakes weeks before they happen by monitoring the amount of metals like zinc and copper in subsoil water near earthquake sites, scientists said Wednesday.",
        "Marine Expedition Finds New Species - Norwegian scientists who explored the deep waters of the Atlantic Ocean said Thursday their findings #151; including what appear to be new species of fish and squid #151; could be used to protect marine ecosystems worldwide."
    ]
})

def create_inference_file(df):
    inference_list = []
    for index, row in df.iterrows():
        content = row['text']
        
        request = {
            "custom_id": f"keyword_extraction-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": 'Extract up to 5 relevant keywords from the given text. Provide only the keywords between double quotes and separated by commas.'},
                    {"role": "user", "content": content}
                ],
            }
        }
        inference_list.append(request)
    return inference_list

def save_inference_file(inference_list):
    filename = f"keyword_extraction_inference_request.jsonl"
    with open(filename, 'w') as file:
        for request in inference_list:
            file.write(json.dumps(request) + '\n')
    return filename

inference_list = create_inference_file(df)
filename = save_inference_file(inference_list)


inference_input_file = client.files.create(
    file=open(filename, "rb"),
    purpose="batch"
)

inference_job = client.batches.create(
    input_file_id=inference_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

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

    updated_job = client.batches.retrieve(inference_job.id)

    if updated_job.status != "completed":
        all_completed = False
        completed = updated_job.request_counts.completed
        total = updated_job.request_counts.total
        output_lines.append(f"Job status: {updated_job.status} - Progress: {completed}/{total}")
    else:
        output_lines.append(f"Job completed!")

    # Clear the output and display updated status
    clear_output(wait=True)
    for line in output_lines:
        display(line)

    if not all_completed:
        time.sleep(10)

job = client.batches.retrieve(inference_job.id)
result_file_id = job.output_file_id
result = client.files.content(result_file_id).content
results = parse_json_objects(result)

for res in results:
    task_id = res['custom_id']
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    text = df.iloc[int(index)]['text']
    print(f'\n -------------------------- \n')
    print(f"Task ID: {task_id}. \n\nINPUT TEXT: {text}\n\nLLM ANSWER: {result}")


