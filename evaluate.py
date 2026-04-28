import os
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

load_dotenv()

#load collection
client=chromadb.PersistentClient(path="./chroma_db")
ef=embedding_functions.DefaultEmbeddingFunction()
collection=client.get_collection(name="finding_aid", embedding_function=ef)
anthropic_client=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#test questions
test_questions = [
    "What types of materials are in the collection?",
    "What dates does the collection cover?",
    "Is the Guillermo Kahlo business card from 1939?", #should say undated
    "Who donated the collection?",
    "Is the 1944 tree and pre columbian head drawing from Frida or Manual", #tests spelling and context
    "Where is the exquisite corpse sketch?", # series 2 drawings box 4, folder 264
    "What day was Frida Kahlo born?", 
    "What did Kahlo eat for breakfast?", # should say it doesn't know
]

results = []

for question in test_questions:
    # retrieve
    retrieved=collection.query(query_texts=[question], n_results=4)
    chunks=retrieved["documents"][0]
    context="\n\n---\n\n".join(chunks)

    # ask claude
    prompt = f"""Answer using ONLY the context below. If the answer isn't there, say so.

Context:
{context}

Question: {question}

Answer:"""
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = message.content[0].text

    # print for manual grading
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"\nA: {answer}")
    print(f"\nRetrieved {len(chunks)} chunks.")
    grade = input("\nGrade retrieval (1=relevant, 0=not relevant): ")
    correct = input("Grade answer (1=correct/honest, 0=incorrect/hallucinated): ")

    results.append({
        "question": question,
        "answer": answer,
        "retrieval": int(grade),
        "correct": int(correct)
    })

# summary
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Question':<45} {'Retrieval':>10} {'Answer':>10}")
print(f"{'-'*65}")
for r in results:
    q = r["question"][:42] + "..." if len(r["question"]) > 42 else r["question"]
    print(f"{q:<45} {r['retrieval']:>10} {r['correct']:>10}")

total=len(results)
retrieval_score=sum(r["retrieval"] for r in results) / total
answer_score=sum(r["correct"] for r in results) / total
print(f"{'-'*65}")
print(f"{'Precision/Recall (retrieval)':<45} {retrieval_score:>10.0%}")
print(f"{'Response accuracy':<45} {answer_score:>10.0%}")