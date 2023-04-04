import os
import glob
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone


app = FastAPI()

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # load variables from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
os.environ['TESSDATA_PREFIX'] = 'tesseract/tessdata'

# Initialize Pinecone vector store
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "langchain1"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vector_db = Pinecone(embeddings, index_name=index_name)

# Define chunk size and maximum number of chunks
chunk_size = 512



@app.post("/add_docs/")
async def add_docs(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        loader = UnstructuredPDFLoader(file.filename)
        data = loader.load()
        print(f'You have {len(data)} document(s) in your data')
        print(f'There are {len(data[0].page_content)} characters in your document')

        # Split text into chunks
        texts = []
        for document in data:
            characters = document.page_content
            chunks = [characters[i:i + chunk_size] for i in range(0, len(characters), chunk_size)]
            texts.extend(chunks)

        # Delete all previous embeddings
        for file in glob.glob("./backend/embeddings-*.pickle"):
            os.remove(file)

        # Embed each chunk and store in file-based database
        embeddings_filename = "./backend/embeddings.pickle"
        embeddings = {}
        for i, text in enumerate(texts):
            # Compute embeddings using Pinecone
            vector_db = Pinecone(index_name=index_name)
            chunk_embeddings = vector_db.embed(text)
            # Save latest embeddings to dictionary
            embeddings[i] = chunk_embeddings

        # Save embeddings dictionary to file
        with open(embeddings_filename, "wb") as f:
            pickle.dump(embeddings, f)

        # Store embeddings in Pinecone vector database
        vector_db = Pinecone(index_name=index_name)
        vector_db.upsert(embeddings)

        return {"message": "Documents added successfully!"}

    except Exception as e:
        print(str(e))
        return {"message": "Error adding documents"}




@app.post("/query/")
async def query(query: str):
    try:
        # Load embeddings from file
        embeddings_filename = f"embeddings-{index_name}.pickle"
        with open(embeddings_filename, "rb") as f:
            embeddings = pickle.load(f)

        # Split query into chunks
        query_chunks = [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

        # Embed query chunks
        query_embeddings = []
        for chunk in query_chunks:
            query_embeddings.append(embeddings[-1].embed(chunk))

        # Compute similarity search using Pinecone
        vector_db = Pinecone(embeddings, index_name=index_name)
        docs = vector_db.similarity_search(query_embeddings, include_metadata=True)

        # Generate response using ChatOpenAI and QA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=500, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        print(response)
        return {"answer": str(response)}

    except Exception as e:
        print(str(e))
        return {"message": "Error with query"}







