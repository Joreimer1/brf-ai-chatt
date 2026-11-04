import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google.antigravity import Agent  # Rättad import utan extra .get


st.set_page_config(page_title="BRF AI-Test", layout="centered")
st.title("🤖 Styrelseassistenten: Prototyp")
st.write("Testa att ställa frågor om föreningens regler.")

# 1. HÄMTA API-NYCKEL FRÅN STREAMLIT SECRETS
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("API-nyckel saknas! Lägg till GEMINI_API_KEY i Advanced Settings -> Secrets.")
    st.stop()

# Sätt nyckeln i systemmiljön så att ChromaDB hittar den automatiskt
os.environ["GEMINI_API_KEY"] = api_key

# Lås sökmodellen till exakt samma som användes vid skapandet av databasen
google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="models/text-embedding-004"
)

@st.cache_resource
def initiera_databas():
    klient = chromadb.PersistentClient(path="./brf_databas")
    samling = klient.get_collection(name="brf_regler", embedding_function=google_ef)
    return samling

try:
    samling = initiera_databas()
except Exception as e:
    st.error(f"Kunde inte ladda databasmappen: {e}")
    st.stop()

# 2. SÄKRAD SÖKFUNKTION (RAG)
def sök_i_dokument(fråga_text):
    try:
        resultat = samling.query(query_texts=[fråga_text], n_results=3)
        # ChromaDB returnerar en nästlad lista, så vi plockar ut det första resultatet [0] säkert
        if resultat and 'documents' in resultat and resultat['documents'] and len(resultat['documents']) > 0:
            textblock = resultat['documents'][0]
            if textblock:
                return "\n\n".join([str(txt) for txt in textblock if txt])
    except Exception as e:
        return f"(Sökningsfel: {e})"
    return "Ingen relevant information hittades i dokumenten."

# 3. CHATT-GRÄNSSNITT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visa tidigare meddelanden
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# När användaren skriver en ny fråga
if fråga := st.chat_input("Skriv din fråga här..."):
    with st.chat_message("user"):
        st.markdown(fråga)
    st.session_state.messages.append({"role": "user", "content": fråga})

    # Gör sökningen i era PDF-filer
    relevant_fakta = sök_i_dokument(fråga)
    
    # Skapa agenten via Google Antigravity SDK
    agent = Agent(
        model="gemini-2.5-flash", 
        system_instructions=(
            f"Du är en BRF-assistent. Svara artigt och sakligt på svenska. "
            f"Basera ditt svar ENBART på denna fakta från föreningens dokument:\n\n{relevant_fakta}\n\n"
            f"Om informationen inte finns i texten ovan, svara ordagrant: "
            f"'Jag hittar ingen information om detta i föreningens gällande dokument. Svaret måste stämmas av med styrelsen.'"
        )
    )
    
    # Generera svaret
    ai_svar = agent.run(fråga)

    # Visa svaret för användaren
    with st.chat_message("assistant"):
        st.markdown(ai_svar)
    st.session_state.messages.append({"role": "assistant", "content": ai_svar})
