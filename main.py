from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about the pizza restaurant.

Here are some reviews of pizza restaurant: {reviews}

Here is a question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n--------------------------------------------------------")
    question = input("Ask a question about pizza restaurant (Q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break
    # Retrieve the most relevant reviews
    reviews = retriever.invoke(question)
    
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)