# from dotenv import load_dotenv
# load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# import streamlit as st

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     )

# st.title("Q&A Bot")
# st.markdown("My  Q&A bot using LangChain and Google Gemini API.")
# query = st.chat_input("Ask a question:")
# if query:
#     st.chat_message("user").markdown(query)
#     response = llm.invoke(query)
#     st.chat_message("assistant").markdown(response.content)

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load environment keys securely
load_dotenv()

# 2. Cache the resource so Streamlit only builds the model connection ONCE
@st.cache_resource
def load_llm_model():
    # Using a highly stable, production-grade free tier model to avoid quota traps
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# Initialize the cached model connection
llm = load_llm_model()

# 3. Build the UI Layout components
st.title("Batuni Bot")
st.markdown("My Batuni Bot using LangChain and Google Gemini API.")

# Initialize the chat message tracker state array if empty
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the separate LangChain model memory array if empty
if "model_history" not in st.session_state:
    st.session_state.model_history = []

# Render the historical conversation log list on screen refresh for the human user
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

query = st.chat_input("Ask a question:")

if query:
    # 1. Display and save the user prompt to the UI list
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    # 2. Append to persistent model memory using LangChain standard tuples
    st.session_state.model_history.append(("human", query))
    
    # 4. Implement safety try/except error block
    try:
        # Generate the request payload using the PERSISTENT session memory
        response = llm.invoke(st.session_state.model_history)
        raw_content = response.content
        
        # --- 🛡️ THE AUTOMATIC SANITIZATION FILTER 🛡️ ---
        if isinstance(raw_content, list) and len(raw_content) > 0:
            item = raw_content[0]
            clean_text = item.get('text', str(item)) if isinstance(item, dict) else getattr(item, 'text', str(item))
        else:
            clean_text = str(raw_content)

        # Delete any trailing signature bracket blocks instantly
        if " [" in clean_text:
            clean_text = clean_text.split(" [")[0].strip()
            
        # 5. Display and save ONLY the clean human-readable text output
        st.chat_message("assistant").markdown(clean_text)
        
        # Save to UI display history
        st.session_state.messages.append({"role": "assistant", "content": clean_text})
        # Save to LangChain model memory history
        st.session_state.model_history.append(("ai", clean_text))
        
    except Exception as e:
        # Gracefully handle temporary API locks without crashing the UI screen
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.error("⚠️ Google's free tier is temporarily busy! Please wait 10 seconds before asking another question.")
        else:
            st.error(f"❌ Connection Error: {e}")



    



# prompts = ChatPromptTemplate.from_messages([
#     ("system", "You are a {role}."),
#     ("user", "{question}"),
# ])


# while True:
#     que = input("Ask a question: ")
#     response = llm.invoke(que)

#     if que.lower() in ["exit", "quit", "bye"]:
#         print("GoodBye.")
#         break

#     print(f"AI Response: {response.content},\n")