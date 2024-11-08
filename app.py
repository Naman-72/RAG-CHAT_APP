import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
from torch import cuda
# from langchain.embeddings.OpenAI import OpenAIEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import faiss

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback

DEVICE = "cuda" if cuda.is_available() else "cpu"
if DEVICE == "cuda":
    st.sidebar.success("🚀 Using GPU")
else:
    st.sidebar.warning("💻 Using CPU - GPU not detected")


with st.sidebar:
    st.title("LLM Chat App 🤗")
    st.markdown('''
                ## About
                This is an LLM Powered RAG ChatBot
                ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Team Crocodile')



def load_vectorstore(store_name):
    """Load the vector store from disk if it exists"""
    try:
        with open(f"faiss_{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
        return vector_store
    except FileNotFoundError:
        return None
    

def create_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def main():
    

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.1,google_api_key=GEMINI_API_KEY)

    pdf = st.file_uploader("Upload Your Pdf",type='pdf')
    # st.write(pdf.name)

    if pdf is not None :
        store_name = pdf.name[:-4]
        vector_store = load_vectorstore(store_name)
        
        if vector_store is None:
            # Process the PDF
            with st.spinner('Processing PDF...'):
                pdf_reader = PdfReader(pdf)
                
                # Extract text
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Create chunks
                chunks = create_chunks(text)


                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl","rb") as f:
                        vector_store = pickle.load(f)
                    st.write("Embeddings Loaded into the Disk")
                else :
                    embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2", model_kwargs={'device': DEVICE})
                    vector_store = faiss.FAISS.from_texts(chunks, embeddings)
                    with open(f"faiss_{store_name}.pkl", "wb") as f:
                        pickle.dump(vector_store, f)
                    st.write("Embeddings Loaded from the Disk")

                st.success('PDF processed successfully!')


        query = st.text_input("Ask a question about your PDF:")
        # st.write(query)

        if query:
            docs = vector_store.similarity_search(query, k=4)
# Gemini 
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=query)
            st.write(response)


# OpenAI
            # llm = OpenAI(temperature=0)
            # chain = load_qa_chain(llm=llm,chain_type="stuff")

            # with get_openai_callback() as cb :
            #     response = chain.run(input_documents=docs,question=query)
            #     print(cb)

            # st.write(response)


if __name__=='__main__':
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("API key not found! Please check your .env file.")

    if not GEMINI_API_KEY:
        st.error("Google API key not found! Please check your .env file.")
    else:
        main()

