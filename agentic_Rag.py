from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import streamlit as st


st.subheader("Agentic Tool")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded =False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "Vectore_store" not in st.session_state:
    st.session_state.vecotr_store = None



def process_document(path):

    laoder = PyPDFDirectoryLoader(path)
    docs = laoder.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 100 ,chunk_overlap = 50)
    doc_split = splitter.split_documents(docs)

    embeddimg = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    vector_store = InMemoryVectorStore.from_documents(
        documents=doc_split,
        embedding=embeddimg
    )

    llm = init_chat_model( 
        model= "groq:openai/gpt-oss-120b"
    )

    @tool
    def retiver_context(query:str):
        """
            This is the tools of understant of the document 
            And pdf of the user they have to answer only in the document's ,
            relatedd question when user asked.
        """

        docs = vector_store.similarity_search(query=query , k=4)
        context = ""
        for doc in docs:
            context += doc.page_content + "\n"

        return context


    system_prompt = """
        You're a Help full assistant that answer the question using retiveral context.
        Always use the "retiveral_tools" for the question reqiureing the external knowledge.
        And if you don't find anything in the document's simpaly reaturn "I didn't find that relative knowledge".
    """


    agent = create_agent(
        model=llm,
        system_prompt=system_prompt,
        tools=[retiver_context],
        checkpointer=st.session_state.memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)




#  upload UI


if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label = "Select_File" , type=["pdf"] , accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./docs_file/"
            for file in uploaded:
                with open(path + file.name,"wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


# chat ui


if st.session_state.document_uploaded and st.session_state.agent:
    query = st.chat_input("Ask anything with Agentic about the document ?")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.history.append({"role":"user","content":query})
        response = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":1}}
        )
        answer  = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.history.append({"role":"ai","content":answer})
