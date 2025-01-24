import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
DATA_FILE = os.path.join(os.getcwd(), "data/eds_data.txt")
# groq_api_key=os.getenv('GROQ_API_KEY')
# # File paths
# DATA_FILE = "data/eds_data.txt"
INDEX_FILE = "models/faiss_index.pkl"
groq_api_key = st.secrets["GROQ_API_KEY"]

# Function to load and split data
def load_and_split_data(file_path, chunk_size=1000, chunk_overlap=50):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Function to build or load FAISS index
def build_or_load_index(docs, index_path="models/faiss_index.pkl", embedding_model="gemma:2b"):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, OllamaEmbeddings(model=embedding_model), allow_dangerous_deserialization=True)

    embedding = OllamaEmbeddings(model=embedding_model)
    vectordb = FAISS.from_documents(docs, embedding)
    vectordb.save_local(index_path)
    return vectordb

# Function to initialize QA chain
def initialize_qa_chain(vectordb, llm_model="Llama-3-Groq-70B-Tool-Use"):
    llm = ChatGroq(model=llm_model, groq_api_key=groq_api_key)
    template = """
Use the following pieces of context to answer the question at the end.

If the question asks for a button, input, table, or any HTML code, provide the relevant information in plain text. Clearly explain how to use the component in EDS or standard HTML with an example.

If the input contains HTML code, identify the EDS-specific equivalent or alternative for the HTML. Provide the answer with examples in EDS-specific syntax.

If the question is not related to components, answer it in plain text using the context provided. 
If you don't know the answer, just say that you don't know and avoid making up an answer.

Use three sentences maximum. Always say "Thanks for asking!" at the end.

Context:
{context}

Question: {question}

Helpful Answer:

"""
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

# Streamlit app
st.write("Loading and processing data...")
docs = load_and_split_data(DATA_FILE)

st.write("Building or loading FAISS index...")
vectordb = build_or_load_index(docs, INDEX_FILE)

st.write("Initializing QA chain...")
qa_chain = initialize_qa_chain(vectordb)

st.title("Document Q&A with Ollama gemma:2b and FAISS")
st.write("Ask questions about the document and get precise answers!")

question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain({"query": question})
            st.write("### Answer:")
            st.write(result.get("result", "No answer found."))

            source_docs = result.get("source_documents", [])
            if source_docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.write(f"#### Document {i + 1}:")
                        st.write(doc.page_content)
            else:
                st.write("No source documents available.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
