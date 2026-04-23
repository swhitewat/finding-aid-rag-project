###
# to run: py -m streamlit run app.py
###

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

load_dotenv()

# load collection
client=chromadb.PersistentClient(path="./chroma_db")
ef=embedding_functions.DefaultEmbeddingFunction()
collection=client.get_collection(name="finding_aid", embedding_function=ef)
anthropic_client=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# page stuff
st.title("final proj")
user_query = st.text_input("describe your questoin:")

if user_query:
    # retrieve chunks
    results=collection.query(query_texts=[user_query], n_results=4)
    chunks=results["documents"][0]
    context="\n\n---\n\n".join(chunks)

    # ask claude
    prompt=f"""Answer using only the context below. If the answer isn't there, say so.
    
Context:
{context}

Question: {user_query}

Answer:"""
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    response=message.content[0].text

    st.write(response)