import streamlit as st
from langchain_groq import ChatGroq

st.set_page_config(page_title='Fitness AI Coach', layout='centered')
st.title('💪 Fitness AI Coach')
st.write('Your Personal Workout & Diet Assistant')

# Sidebar
with st.sidebar:
    st.header('⚙️ Configuration')
    user_api_key = st.text_input('Enter Groq API Key:', type='password')
    st.info('Get workout plans, diet tips, and fitness guidance')

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input
if user_query := st.chat_input('Ask your fitness question...'):

    if not user_api_key:
        st.error("Please enter your API Key in the sidebar first!")

    else:
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        with st.chat_message('user'):
            st.markdown(user_query)

        llm = ChatGroq(
            temperature=0.5,
            model='llama-3.3-70b-versatile',
            api_key=user_api_key
        )

        system_prompt = """
        You are a certified Fitness Coach AI.

        You help users with:
        - Workout plans
        - Fat loss programs
        - Muscle gain programs
        - Diet and nutrition tips
        - Beginner fitness guidance
        - Weekly training schedules

        Keep answers practical and easy to follow.
        """

        with st.spinner('Coach is preparing your plan...'):
            response = llm.invoke(system_prompt + "\nUser: " + user_query)
            bot_answer = response.content

        st.session_state.messages.append({'role': 'assistant', 'content': bot_answer})
        with st.chat_message('assistant'):
            st.markdown(bot_answer)
