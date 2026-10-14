from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import streamlit as st

llm = ChatGroq(
     model="openai/gpt-oss-120b",
     streaming=True
)

search = GoogleSerperAPIWrapper()
thread_config = {"configurable": {"thread_id": "1"}}

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []





agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a search engine",
    checkpointer=st.session_state.memory
)

st.subheader("Chat with gpt ")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)
 


query = st.chat_input("Ask anything with gpt ?")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    respose = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        thread_config,
        stream_mode="messages"
        )

    ai_contener = st.chat_message("ai")

    with ai_contener:
        space = st.empty()

        message = ""

        for chuks in respose:
            message = message + chuks[0].content
            space.write(message)


        st.session_state.history.append({"role":"ai","content":message})








