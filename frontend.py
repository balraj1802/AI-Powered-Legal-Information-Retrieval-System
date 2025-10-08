from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Setup LLM
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")
critic_model = ChatGroq(model="llama-3.1-8b-instant")  # Example: use a different model for critique

# Step 2: Retrieve Docs
def retrieve_docs(query, k=3):
    return faiss_db.similarity_search(query, k=k)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Step 3: Answer Question
custom_prompt_template = """
You are an AI lawyer. Use only the pieces of information provided in the context to answer the user's question. 
If the answer is not in the context, say "I don't know". 
Do NOT make up an answer or use outside knowledge.

Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(custom_prompt_template)

def answer_query(documents, model1, query, model2=None):
    context = get_context(documents)
    chain = prompt | model1
    answer = chain.invoke({"question": query, "context": context})
    # Optionally use critic_model for further critique
    if model2:
        critique_prompt = ChatPromptTemplate.from_template(
            "Critique the following answer as an expert legal reviewer:\n\nAnswer: {answer}\n"
        )
        critique_chain = critique_prompt | model2
        critique = critique_chain.invoke({"answer": answer})
        return {"answer": answer, "critique": critique}
    return {"answer": answer}

# Example Run
if __name__ == "__main__":
    question = "If a government forbids the right to assemble peacefully which articles are violated and why?"
    retrieved_docs = retrieve_docs(question)
    print("AI Lawyer:", answer_query(retrieved_docs, llm_model, question))
    response = answer_query(documents=retrieved_docs, model1=llm_model, query=user_query, model2=critic_model)
