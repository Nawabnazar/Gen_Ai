from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import SQLDatabase,GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


import streamlit as st

db = SQLDatabase.from_uri("sqlite:///My_task.db")

db.run(
    """
   CREATE TABLE IF NOT EXISTS task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
)



search = GoogleSerperAPIWrapper()




llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

toolkit = SQLDatabaseToolkit(db = db , llm=llm)
tools = toolkit.get_tools()




system_prompt = """
You are a task management assistant that interacts with a SQL database conataining a 'task ' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.

CRUD OPERATIONS:
CREATE: INSERT INTO tasks(title, description, status)
READ: SELECT * FROM tasks WHERE ... LIMIT 10
UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/progress/completed), created_at.
"""
@st.cache_resource
def get_agent():
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )
    return agent

agent = get_agent()


st.subheader("Task_agent - Make you're task with Agent")

if "messages" not in st.session_state:
      st.session_state.messages = []

for message in st.session_state.messages:
      st.chat_message(message["role"]).markdown(message["content"])

      

prompt = st.chat_input("Ask me about task ?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("ai"):
      with st.spinner("processing..."):
                    res = agent.invoke(
                     {"messages":[{
                         "role":"user" , "content": prompt
                     }]},
                    {"configurable": {"thread_id": "1"}}
                     )
                    result = res["messages"][-1].content
                    st.markdown(result)
                    st.session_state.messages.append({"role":"ai","content":result})
        

