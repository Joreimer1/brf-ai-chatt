import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google_antigravity import Agent

st.set_page_config(page_title="BRF AI-Test", layout="centered")
st.title("🤖 Styrelseassistenten: Prototyp")
st.write("Testa att ställa frågor om föreningens regler.")

# Vi hämtar API-nyckeln från Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("API-nyckel saknas! Lägg till GEMINI_API_KEY i Advanced Settings -> Secrets.")
    st.stop()

# Ny funktion: Vi ber Google räkna ut sökvektorerna istället för lokala datorn
google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    api_key=api_key
)

@st.cache_resource
def initiera_databas():
    klient = chromadb.PersistentClient(path="./brf_databas")
    # Vi säger till databasen att använda Google istället för sentence-transformers
    samling = klient.get_collection(name="brf_regler", embedding_function=google_ef)
    return samling

try:
    samling = initiera_databas()
except Exception as e:
    st.error(f"Kunde inte ladda databasmappen: {e}")
    st.stop()

def sök_i_dokument(fråga_text):
    resultat = samling.query(query_texts=[fråga_text], n_results=3)
    return "\n\n".join(resultat['documents'])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if fråga := st.chat_input("Skriv din fråga här..."):
    with st.chat_message("user"):
        st.markdown(fråga)
    st.session_state.messages.append({"role": "user", "content": fråga})

    relevant_fakta = sök_i_dokument(fråga)
    
    agent = Agent(
        model="gemini-2.5-flash", 
        system_instructions=f"Du är en BRF-assistent. Svara artigt på svenska. Basera ditt svar ENBART på denna fakta: {relevant_fakta}. Om infon saknas, säg att styrelsen måste kontaktas."
    )
    
    ai_svar = agent.run(fråga)

    with st.chat_message("assistant"):
        st.markdown(ai_svar)
    st.session_state.messages.append({"role": "assistant", "content": ai_svar})
