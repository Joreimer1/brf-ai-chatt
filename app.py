import streamlit as st
import chromadb
from google_antigravity import Agent

st.set_page_config(page_title="BRF AI-Test", layout="centered")
st.title("🤖 Styrelseassistenten: Prototyp")
st.write("Testa att ställa frågor om föreningens regler så ser du hur snabbt AI:n svarar utifrån våra 170 PDF-dokument.")

# 1. Starta databasen i molnet (Hämtar data från mappen du laddar upp)
@st.cache_resource
def initiera_databas():
    klient = chromadb.PersistentClient(path="./brf_databas")
    samling = klient.get_collection(name="brf_regler")
    return samling

try:
    samling = initiera_databas()
except Exception:
    st.error("Kunde inte ladda databasmappen. Se till att 'brf_databas' ligger i samma mapp på GitHub.")
    st.stop()

# 2. Sökfunktion (RAG)
def sök_i_dokument(fråga_text):
    resultat = samling.query(query_texts=[fråga_text], n_results=3)
    return "\n\n".join(resultat['documents'])

# 3. Enkel Chatt-loop
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if fråga := st.chat_input("Skriv din fråga här (t.ex: Vad gäller för borrning?)"):
    with st.chat_message("user"):
        st.markdown(fråga)
    st.session_state.messages.append({"role": "user", "content": fråga})

    # Gör sökningen och kör AI:n
    relevant_fakta = sök_i_dokument(fråga)
    
    # Här skapar vi agenten. Den använder din API-nyckel i molnet automatiskt.
    agent = Agent(
        model="gemini-2.5-flash", 
        system_instructions=f"Du är en BRF-assistent. Svara artigt på svenska. Basera ditt svar ENBART på denna fakta: {relevant_fakta}. Om infon saknas, säg att styrelsen måste kontaktas."
    )
    
    ai_svar = agent.run(fråga)

    with st.chat_message("assistant"):
        st.markdown(ai_svar)
    st.session_state.messages.append({"role": "assistant", "content": ai_svar})
