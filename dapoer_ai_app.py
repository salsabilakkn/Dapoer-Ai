import streamlit as st
from dapoer_module import create_agent

st.set_page_config(
    page_title="🍲 Dapoer-AI",
    page_icon="🍛"
)

st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")


api_key = st.text_input(
    "🔑 Masukkan API Key Gemini kamu:",
    type="password"
)

if not api_key:
    st.warning("Silakan masukkan API Key untuk mulai.")
    st.stop()


@st.cache_resource
def load_agent(key):
    return create_agent(key)


agent = load_agent(api_key)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hai! Mau masak apa hari ini?"
        }
    ]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input(
    "Tanyakan resep, bahan, metode..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)

        except Exception as e:
            response = f"⚠️ Error: {e}"

        st.markdown(response)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
