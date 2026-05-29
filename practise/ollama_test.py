import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {
            'role': 'user',
            'content': 'What is Machine Learning?'
        }
    ]
)

print(response['message']['content'])
