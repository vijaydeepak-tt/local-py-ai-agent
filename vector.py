from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB location
db_location = "./chroma_db"

# Check if the database already exists
add_to_db = not os.path.exists(db_location)

if add_to_db:
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={
                "Title": row["Title"],
                "Review": row["Review"],
                "Rating": row["Rating"],
                "Date": row["Date"],
            },
            id=str(i),
        )
        documents.append(doc)
        ids.append(str(i))
    
# Create the vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=db_location,
    collection_name="restaurant_reviews",
)

# Add documents to the vector store
if add_to_db:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
