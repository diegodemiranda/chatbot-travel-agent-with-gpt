from openai import OpenAI
from config import api_key

client = OpenAI(api_key=api_key)

messages = [
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex concepts."}
]

input_message = input()
messages.append({"role": "user", "content": input_message})

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=1
)
# format the answer
answer = response['choices'][0]['message']['content']
# create a chat history
messages.append({"role": "system", "content": answer})

print(answer)
