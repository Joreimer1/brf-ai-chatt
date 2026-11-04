import os
import asyncio
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google.antigravity import Agent, LocalAgentConfig

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
    
    # Skapa konfigurationen för agenten enligt Google Antigravity SDK-standard
    konfiguration = LocalAgentConfig(
        system_instructions=(
            f"Du är en BRF-assistent. Svara artigt och sakligt på svenska. "
            f"Basera ditt svar ENBART på denna fakta från föreningens dokument:\n\n{relevant_fakta}\n\n"
            f"Om informationen inte finns i texten ovan, svara ordagrant: "
            f"'Jag hittar ingen information om detta i föreningens gällande dokument. Svaret måste stämmas av med styrelsen.'"
        )
    )
    
    # Starta agenten med konfigurationen
    agent = Agent(config=konfiguration)

    # Funktion för att köra agenten asynkront enligt SDK-specifikationen
    async def kör_agent():
        async with agent:
            return await agent.chat(fråga)
    
       # Funktion för att köra agenten och packa upp texten asynkront enligt SDK-specifikationen
    async def hämta_svar():
        async with agent:
            svar_objekt = await agent.chat(fråga)
            return await svar_objekt.text()  # Här väntar vi in den rena texten säkert!
    
    # Kör den asynkrona loopen inuti Streamlit
    ai_text = asyncio.run(hämta_svar())

    # Visa svaret för användaren
    with st.chat_message("assistant"):
        st.markdown(ai_text)
    st.session_state.messages.append({"role": "assistant", "content": ai_text})
