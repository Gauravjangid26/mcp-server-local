# app.py

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Note: corrected import
from mcp_use import MCPAgent, MCPClient

# Load environment variables
load_dotenv()
grok_api_key = os.getenv("GROQ_API_KEY")

if not grok_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it in your .env file.")
    st.stop()

# Title and description
st.set_page_config(page_title="MCP Agent Chat", layout="centered")
st.title("🧠 MCP Agent Chat")
st.write("Chat with an intelligent agent backed by LangChain, Groq LLM, and MCP tools.")

# Function to initialize the agent and client
@st.cache_resource(show_spinner=False)
def initialize_agent():
    config_path = "browse_mcp.json"
    client = MCPClient.from_config_file(config_path)
    llm = ChatGroq(model="qwen-qwq-32b", api_key=grok_api_key)  # Change model if needed
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )
    return client, agent

# Initialize and store in session state
if "client" not in st.session_state or "agent" not in st.session_state:
    st.session_state.client, st.session_state.agent = initialize_agent()
    st.session_state.chat_history = []

agent = st.session_state.agent
client = st.session_state.client
chat_history = st.session_state.chat_history

# Chat history display
for msg in chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Clear and exit controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation"):
        agent.clear_conversation_history()
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    if st.button("🚪 Exit Session"):
        asyncio.run(client.close_all_sessions())
        st.session_state.clear()
        st.experimental_rerun()

# Chat input
user_input = st.chat_input("Ask me something...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Assistant is thinking..."):
        try:
            response = asyncio.run(agent.run(user_input))
            st.chat_message("assistant").markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.chat_message("assistant").markdown(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
