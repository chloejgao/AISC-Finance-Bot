import streamlit as st
import pandas as pd
import anthropic
import os

from dotenv import load_dotenv
load_dotenv()

# load reuters & stock data
try:
    reuters_df = pd.read_csv("reuters_all_news_2025-05-09.csv")
    stock_df = pd.read_csv("stock_data_1month.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def format_context(question):
    # top headlines
    recent_news = "\n".join(reuters_df["headline"].dropna().astype(str).tolist()[:10])

    # flatten stock headers
    stock_df.columns = [
        "_".join([str(c) for c in col]).strip() if isinstance(col, tuple) else str(col)
        for col in stock_df.columns
    ]

    # summarize close and volume headers
    summary_lines = []
    for col in stock_df.columns:
        if "Close" in col or "Volume" in col:
            try:
                avg = pd.to_numeric(stock_df[col], errors='coerce').dropna().mean()
                summary_lines.append(f"{col}: avg = {avg:.2f}")
            except:
                continue

    stock_summary = "\n".join(summary_lines)

    # build Claude prompt
    context = f"""
You are a financial assistant. Based on the market question, recent Reuters headlines, and 1-month stock data summary, give high-level investment advice to an average investor.

🔍 Market Question:
{question}

📰 Top Reuters Headlines:
{recent_news}

📊 Stock Summary (average prices and volumes):
{stock_summary}
"""
    return context

st.set_page_config(page_title="How Powerful Was That News?")
st.title("🗞️ How Powerful Was That News?")
st.write("Ask a market question. The bot will use Reuters news and stock data to give general investment advice.")

user_question = st.text_input("Ask a market question:", placeholder="e.g., What should investors do after the Fed’s last rate decision?")

# Claude
if user_question:
    with st.spinner("Thinking..."):
        prompt = format_context(user_question)

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                answer = response.content[0].text
                st.markdown("### 🧠 General Advice")
                st.markdown(answer)
            except Exception as e:
                st.error(f"⚠️ Could not extract Claude's response: {e}")

        except Exception as e:
            st.error(f"❌ API Error: {e}")
