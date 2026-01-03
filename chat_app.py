import streamlit as st
import anthropic

API_KEY = "Api-key"
client = anthropic.Anthropic(api_key=API_KEY)

st.set_page_config(page_title="Claude Chatbot", page_icon="🤖")
st.title("Claude Chatbot")
st.markdown("Ask anything and Claude will respond using the Anthropic API.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Claude"
    st.markdown(f"**{role}:** {msg['content']}")

user_input = st.text_input("Your message:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=st.session_state.messages
        )

        reply = response.content[0].text.strip()
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error: {e}")
