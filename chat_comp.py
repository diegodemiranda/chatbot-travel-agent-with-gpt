from openai import OpenAI
from config import api_key

client = OpenAI(api_key=api_key)

messages=[
  {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts."},
  {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages,
  temperature=1
)

print(response.choices[0].message)
