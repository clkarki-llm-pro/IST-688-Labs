import streamlit as st
from openai import OpenAI

# The system prompt shapes how the bot behaves on every turn
SYSTEM_PROMPT = """You are a friendly assistant explaining things to a 10 year old child.

Rules for every answer:
- Use simple words and short sentences. No jargon. If you must use a hard word,
  explain it right away.
- Answer the user's question first, briefly.
- Then always end your message by asking exactly: "Do you want more info?"

If the user replies "yes" (or anything meaning yes), give more detail about the
same topic you were just discussing, still in simple language, and then ask
"Do you want more info?" again.

If the user replies "no" (or anything meaning no), do not add more detail.
Instead reply with something like "Okay! What else can I help you with?"
and wait for a new question."""

#Show title
st.title("My Lab 3 Question Answering Chatbot")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

#Create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = \
    [{"role": "assistant", "content": "How can I help you?"}]

#Display chat messages from history on app rerun
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# React to user input
if prompt := st.chat_input("What is up?"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the streaming response from the LLM
    client = st.session_state.client
    # Build the buffer: last two user messages and the replies to them
    max_user_messages = 2
    buffer = []
    user_count = 0
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            user_count += 1
            if user_count > max_user_messages:
                break
        buffer.append(msg)
    buffer.reverse()

    stream = client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + buffer,
        stream=True
    )

    # Stream the assistant response into the chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})