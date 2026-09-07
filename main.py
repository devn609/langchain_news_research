import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pickle
import time
#from langchain import OpenAI
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains import RetrievalQAWithSourcesChain
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader

#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from pathlib import Path

#...OpenAI OpenAIEmbeddings RecursiveCharacterTextSplitter RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
#file_path = DATA_DIR / "faiss_store.pkl"

file_path = DATA_DIR / "faiss_store_local"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    
    docs = text_splitter.split_documents(data)
    print(f"Loaded {len(docs)} documents.")
    if not docs:
        print(f"The 'docs' list is empty")
        raise ValueError("The 'docs' list is empty. Ensure your loader or splitter returned content.")
    
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    # with open(file_path, "wb") as f:
    #    pickle.dump(vectorstore_openai, f)

    vectorstore_openai.save_local(file_path)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
                    file_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
