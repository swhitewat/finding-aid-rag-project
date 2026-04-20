import os
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

load_dotenv()

# loading vector
def load_collection():
    client=chromadb.PersistentClient(path="./chroma_db")
    ef=embedding_functions.DefaultEmbeddingFunction()
    collection=client.get_collection(
        name="finding_aid",
        embedding_function=ef
    )
    return collection

# get chunks
def retrieve(collection, query, n_results=4):
    results=collection.query(
        query_texts=[query],
        n_results=n_results
    )
    chunks=results["documents"][0]
    return chunks

# claude answer
def ask_claude(query, chunks):
    client=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    context="\n\n---\n\n".join(chunks)

    prompt = f"""You are an assistant that helps users explore the Frida Kahlo Papers 
finding aid from the National Museum of Women in the Arts.

Answer the user's question using ONLY the context provided below. 
If the answer is not in the context, say so. Do not guess or make things up.
If attribution of a piece is uncertain, reflect that uncertainty in your answer. Do not use
special formatting in output.

Context:
{context}

Question: {query}

Answer:"""

    message=client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

# main
if __name__=="__main__":
    print("Frida Kahlo Finding Aid - RAG Query System")
    print("   Type question below. Type 'quit' to exit.\n")

    collection=load_collection()

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye")
            break
        if not query:
            continue

        print("\n Searching...\n")
        chunks=retrieve(collection, query)
        answer=ask_claude(query, chunks)

        print(f" Answer:\n{answer}\n")
        print("-" * 60 + "\n")