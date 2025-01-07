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
        "It hums, crackles, and I think I'm having problems with my equipment. As soon as I use any of my other cords then the problem is gone. Hosa makes some other products that have good value. But based on my experience I don't recommend this one.",
        "I bought this to use with my keyboard. I wasn't really aware that there were other options for keyboard pedals. It doesn't work as smoothly as the pedals do on an acoustic piano, which is what I'd always used. Doesn't have the same feel either. Nowhere close.In my opinion, a sustain pedal like the M-Audio SP-2 Sustain Pedal with Piano Style Action or other similar pedal is a much better choice. The price difference is only a few dollars and the feel and action are so much better.",
        "This cable disproves the notion that you get what you pay for. It's quality outweighs its price. Let's face it, a cable is a cable is a cable. But the quality of these cables can vary greatly. I replaced a lighter cable with this one and I was surprised at the difference in the quality of the sound from my amp. I have an Ibanez ART series guitar into an Ibanez 15 watt amp set up in my home. With nothing changed but the cable, there was a significant difference in quality and volume. So much so that I checked with my guitar teacher who said he was not surprised. The quality appears good. The ends are heavy duty and the little bit of hum I had due to the proximity of everything was attenuated to the point where it was inconsequential. I've seen more expensive cables and this one is (so far) great.Hosa GTR210 Guitar Cable 10 Ft",
        "Bought this to hook up a Beta 58 to a Panasonic G2 DSLR and a Kodak Zi8 for interviews. Works the way it's supposed to. 90 degree TRS is a nice touch. Good price.",
        "96	Just received this cord and it seems to work as expected. What can you say about an adapter cord? It is well made, good construction and sound from my DSLR with my mic is superb."
    ]
})

def create_inference_file(df):
    inference_list = []
    for index, row in df.iterrows():
        content = row['text']
        
        request = {
            "custom_id": f"sentiment-analysis-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": 'Analyze the sentiment of this text and respond with one word: positive, negative, or neutral.'},
                    {"role": "user", "content": content}
                ],
            }
        }
        inference_list.append(request)
    return inference_list

def save_inference_file(inference_list):
    filename = f"sentiment_analysis_inference_request.jsonl"
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
    print(f"Task ID: {task_id}. \n\nINPUT TEXT: {text}\n\nLLM OUTPUT: {result}")
