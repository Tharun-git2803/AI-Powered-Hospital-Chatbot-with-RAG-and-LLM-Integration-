import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Health Assistant")
st.title("🩺 AI Hospital Assistant (Hybrid RAG System)")

# =========================
# API KEY
# =========================
with open("grokapi.txt", "r") as f:
    api_key = f.read().strip()

# =========================
# LLM
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0.3
)

# =========================
# EMBEDDINGS + DB LOAD
# =========================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# =========================
# MEMORY
# =========================
if "memory" not in st.session_state:
    st.session_state.memory = InMemoryChatMessageHistory()

memory = st.session_state.memory

# =========================
# PROMPTS
# =========================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hospital assistant.\n"
     "Answer ONLY from hospital context.\n"
     "If not found say: Not available in records.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a safe medical assistant.  answer the questions in  short from and dont revel ur details. for generic quetions give generic answers example like for hello hey hello if he say his name add name and ask how can i help u "
     ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# =========================
# FUNCTIONS
# =========================

def retrieve_docs(question):
    return vectorstore.similarity_search_with_score(question, k=3)


def should_use_rag(question, docs):

    if not docs:
        return False

    top_score = docs[0][1]

    keywords = ["doctor", "hospital", "timing", "location", "clinic"]

    if any(k in question.lower() for k in keywords):
        return True

    return top_score < 1.5


def rag_answer(question, docs):

    context = "\n\n".join([d.page_content for d, _ in docs])

    prompt = rag_prompt.format(
        context=context,
        question=question
    )

    return llm.invoke(prompt).content


def chat_answer(question):

    prompt = chat_prompt.format(
        chat_history=memory.messages,
        question=question
    )

    return llm.invoke(prompt).content


def get_answer(question):

    docs = retrieve_docs(question)

    use_rag = should_use_rag(question, docs)

    if use_rag:
        answer = rag_answer(question, docs)
        source = "📚 RAG (Hospital Data)"
    else:
        answer = chat_answer(question)
        source = "🤖 LLM Chatbot"

    memory.add_message(HumanMessage(content=question))
    memory.add_message(AIMessage(content=answer))

    return answer, source, docs


# =========================
# CHAT UI
# =========================
for msg in memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

# =========================
# INPUT
# =========================
query = st.chat_input("Ask your medical question...")

if query:

    with st.chat_message("user"):
        st.write(query)

    answer, source, docs = get_answer(query)

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(source)

    # DEBUG
    with st.expander("🔍 Retrieved Docs"):
        for doc, score in docs:
            st.write(score)
            st.write(doc.page_content[:200])
            st.write("---")