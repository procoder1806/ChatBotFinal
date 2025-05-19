import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

# === Load Environment Variables (HF Spaces uses Secrets) ===
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

# === Streamlit Page Config ===
st.set_page_config(page_title="⚡ Groq Chatbot", layout="centered")
st.title("🤖 Chatbot (Groq + LangChain + Hugging Face Spaces)")

# === Sidebar Model Selector ===
AVAILABLE_MODELS = {
    "LLaMA 3 (8B)": "llama3-8b-8192",
    "LLaMA 3.3 (70B)": "llama-3.3-70b-versatile",
    "Gemma (9B)": "gemma2-9b-it"
}

selected_model_label = st.sidebar.selectbox("Choose a model", list(AVAILABLE_MODELS.keys()))
selected_model = AVAILABLE_MODELS[selected_model_label]
st.sidebar.caption(f"💡 Current model: `{selected_model}`")

# === Session State Setup ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "last_model" not in st.session_state:
    st.session_state.last_model = selected_model

# === Reset Memory if Model Changed ===
if st.session_state.last_model != selected_model:
    st.session_state.memory.clear()
    st.session_state.last_model = selected_model

# === LangChain LLM Setup ===
llm = ChatOpenAI(
    model=selected_model,
    temperature=0.7,
    openai_api_key=api_key,
    openai_api_base=api_base
)

# === LangChain Prompt & Chain ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()

# === Chat UI ===
user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    memory = st.session_state.memory
    history = memory.load_memory_variables({})["history"]
    response = chain.invoke({"input": user_input, "history": history})
    memory.save_context({"input": user_input}, {"output": response})

# === Display Chat History ===
for msg in st.session_state.memory.chat_memory.messages:
    role = "🧑 You" if msg.type == "human" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg.content}")
