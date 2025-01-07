#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


# In[ ]:




# In[7]:


from openai import OpenAI

import pandas as pd
import time
import json
from IPython.display import clear_output, display


# In[19]:


# Set up the client
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)


# In[20]:


df = pd.DataFrame({
    "text": [
        "Breakfast at Tiffany's: A young New York socialite becomes interested in a young man who has moved into her apartment building, but her past threatens to get in the way.",
        "Giant: Sprawling epic covering the life of a Texas cattle rancher and his family and associates.",
        "From Here to Eternity: In Hawaii in 1941, a private is cruelly punished for not boxing on his unit's team, while his captain's wife and second-in-command are falling in love.",
        "Lifeboat: Several survivors of a torpedoed merchant ship in World War II find themselves in the same lifeboat with one of the crew members of the U-boat that sank their ship.",
        "The 39 Steps: A man in London tries to help a counter-espionage Agent. But when the Agent is killed, and the man stands accused, he must go on the run to save himself and stop a spy ring which is trying to steal top secret information."
    ]
})


# In[21]:


def create_inference_file(df):
    inference_list = []
    for index, row in df.iterrows():
        content = row['text']
        
        request = {
            "custom_id": f"movie_classification-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": 'Classify the main genre of the given movie description based on the following genres(Respond with only the genre): “Action”, “Adventure”, “Comedy”, “Crime”, “Documentary”, “Drama”, “Fantasy”, “Horror”, “Romance”, “Sci-Fi”.'},
                    {"role": "user", "content": content}
                ],
            }
        }
        inference_list.append(request)
    return inference_list

def save_inference_file(inference_list):
    filename = f"movie_classification_inference_request.jsonl"
    with open(filename, 'w') as file:
        for request in inference_list:
            file.write(json.dumps(request) + '\n')
    return filename


# In[12]:


inference_list = create_inference_file(df)
filename = save_inference_file(inference_list)


# In[22]:


get_ipython().system('head -n 1 movie_classification_inference_request.jsonl')


# In[23]:


inference_input_file = client.files.create(
    file=open(filename, "rb"),
    purpose="batch"
)


# In[24]:


inference_job = client.batches.create(
    input_file_id=inference_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)


# In[25]:


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


# In[26]:


job = client.batches.retrieve(inference_job.id)
result_file_id = job.output_file_id
result = client.files.content(result_file_id).content
parse_json_objects(result)


# In[ ]:




