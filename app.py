import os
import streamlit as st
import time
from dotenv import load_dotenv

# Imports
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from pathlib import Path
load_dotenv()

def process_urls_data(llm, file_path, urls, main_placeholder, main_msg):
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text(f"Text Splitter...Started... {len(data)} docs")
    docs = text_splitter.split_documents(data)
    print(f"Loaded {len(docs)} documents.")
    if not docs:
        print(f"The 'docs' list is empty")

    main_msg.text(f"Text Splitter...after split: {len(docs)} chunks")

    # Create embeddings and save
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Building...✅")

    # Save locally (Recommended over pickle.dump)
    vectorstore_openai.save_local(file_path)
    time.sleep(2)


def load_faiss_store_and_query(llm, file_path, query):
    if os.path.exists(file_path):
        embeddings = OpenAIEmbeddings()
        # IMPORTANT: allow_dangerous_deserialization=True is required for local FAISS loads
        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        # In newer LangChain, invoke() is preferred over calling the object directly
        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)


# Main Streamlit UI
def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    file_path = DATA_DIR / "faiss_store_local"

    st.title("Document Research Tool")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")

    main_placeholder = st.empty()
    main_msg = st.empty()

    # Use newer OpenAI class from langchain_openai
    llm = OpenAI(temperature=0.9, max_tokens=500)

    if process_url_clicked:
        # Filter out empty URLs
        valid_urls = [u for u in urls if u.strip()]
        if valid_urls:
            process_urls_data(llm, file_path, valid_urls, main_placeholder, main_msg)
        else:
            st.error("Please enter at least one URL.")

    query = main_placeholder.text_input("Question: ")
    if query:
        load_faiss_store_and_query(llm, file_path, query)


if __name__ == "__main__":
    main()