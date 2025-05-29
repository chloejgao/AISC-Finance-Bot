import anthropic

client = anthropic.Anthropic(api_key="API-KEY")

print("Claude Chatbot (type 'exit' to quit)")
history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        messages=history
    )

    reply = response.content[0].text.strip()
    print("Claude:", reply)
    history.append({"role": "assistant", "content": reply})
