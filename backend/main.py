from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
# try to cut text size more - summarize
# from langchain.chains.summarize import load_summarize_chain
# from langchain import OpenAI


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

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "langchain1"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.post("/add_docs/")
async def add_docs(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        loader = UnstructuredPDFLoader(file.filename)
        data = loader.load()
        print(f'You have {len(data)} document(s) in your data')
        print(f'There are {len(data[0].page_content)} characters in your document')

        # Summarize parts (제대로 작동하지 않으므로 다음에 다시 시도.)
        # llm = OpenAI(openai_api_key=OPENAI_API_KEY)
        # model = load_summarize_chain(llm=llm, chain_type="stuff")
        # model.run(data)
        # print(f'[After summarization] You have {len(data)} document(s) in your data')
        # print(f'[After summarization] There are {len(data[0].page_content)} characters in your document')

        # chunk size 1000 -> 500 -> 250 (250 정도로 줄여야 대답에 대한 토큰량이 보존되므로 할만함.)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        print(f'Now you have {len(texts)} documents')
        global docsearch
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
        return {"message": "Documents added successfully!"}
    except Exception as e:
        print(str(e))
        return {"message": "Error adding documents"}


@app.post("/query/")
async def query(query: str):
    try:
        # postfix = "Answer in Korean"
        query_with_postfix = query
        # query_with_postfix = query + " " + postfix
        docs = docsearch.similarity_search(query_with_postfix, include_metadata=True)
        # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=500, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query_with_postfix)
        print(response)
        return {"answer": str(response)}

    except Exception as e:
        print(str(e))
        return {"message": "Error with query"}


