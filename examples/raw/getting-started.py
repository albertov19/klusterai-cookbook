#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the API_KEY environment variable.")


# In[15]:


from openai import OpenAI
import json
client = OpenAI(
    base_url="https://api.kluster.ai/v1",  
    api_key=api_key, # Replace with your actual API key
)

tasks = [{
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Argentina?"},
            ],
            "max_tokens": 1000,
        },
    },
    {
        "custom_id": "request-2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
            "messages": [
                {"role": "system", "content": "You are a maths tutor."},
                {"role": "user", "content": "Explain the Pythagorean theorem."},
            ],
            "max_tokens": 1000,
        },
    }
    # Additional tasks can be added here
]

# Save tasks to a JSONL file (newline-delimited JSON)
file_name = "my_inference_test.jsonl"
with open(file_name, "w") as file:
    for task in tasks:
        file.write(json.dumps(task) + "\n")


# In[16]:


inference_input_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)

inference_input_file.to_dict()


# In[17]:


inference_request = client.batches.create(
    input_file_id=inference_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

inference_request.to_dict()


# In[18]:


import time

# Poll the job's status until it's complete
while True:
    inference_status = client.batches.retrieve(inference_request.id)
    print("Job status: {}".format(inference_status.status))
    print(
        f"Completed tasks: {inference_status.request_counts.completed} / {inference_status.request_counts.total}"
    )

    if inference_status.status.lower() in ["completed", "failed", "cancelled"]:
        break

    time.sleep(10)  # Wait for 10 seconds before checking again

inference_status.to_dict()


# In[19]:


# Check if the job completed successfully
if inference_status.status.lower() == "completed":
    # Retrieve the results
    result_file_id = inference_status.output_file_id
    results = client.files.content(result_file_id).content

    # Save results to a file
    result_file_name = "inference_results.jsonl"
    with open(result_file_name, "wb") as file:
        file.write(results)
    print(f"Results saved to {result_file_name}")
else:
    print(f"Job failed with status: {inference_status.status}")


# In[ ]:


client.batches.list(limit=2).to_dict()


# In[ ]:


client.batches.cancel(inference_request.id)


# In[ ]:


client.models.list().to_dict()

