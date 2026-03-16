import streamlit as st
from langchain_groq import ChatGroq

st.set_page_config(page_title='My AI Chat', layout='centered')
st.title('🤖 The Groq Chatbot')
st.write('A fully integrated, memory enabled AI Assistant')

# The Sidebar
with st.sidebar:
    st.header('⚙️Configuration')
    user_api_key = st.text_input('Enter Groq API Key:', type='password')
    st.info('Your key is required to wake the AI Brain')

# The Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']): # Automatically display different emojis for user and assistant messages
        st.markdown(msg['content'])

if user_query := st.chat_input('Message the AI....'):

    if not user_api_key:
        st.error("Please enter your API Key in the sidebar first!")

    else:
        st.session_state.messages.append({'role':'user', 'content':user_query})
        with st.chat_message('user'):
            st.markdown(user_query)

        llm = ChatGroq(
            temperature=0.7,
            model = 'llama-3.3-70b-versatile',
            api_key = user_api_key
        )

        with st.spinner('AI is thinking...'):
            response = llm.invoke(st.session_state.messages)
            bot_answer = response.content

        st.session_state.messages.append({'role':'assistant', 'content':bot_answer})
        with st.chat_message('assistant'):
            st.markdown(bot_answer)
