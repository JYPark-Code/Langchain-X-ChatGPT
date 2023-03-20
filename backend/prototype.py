from dotenv import load_dotenv
import os
# Langchain Requirement module
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Pinecone Requirement module
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
# Query those docs to get your answer back
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
# implement tesseract
os.environ['TESSDATA_PREFIX'] = 'tesseract/tessdata'


load_dotenv()  # load variables from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# Load your data

loader = UnstructuredPDFLoader("docs/27V5_BK_manual_for_KR.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

data = loader.load()

print(f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document')


# Chunk your data up into smaller documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print(f'Now you have {len(texts)} documents')


embbedings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#initialize pinecone

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "langchain1"

docsearch = Pinecone.from_texts([t.page_content for t in texts ], embbedings, index_name=index_name)

# pinecone only
# query = "차량의 전장은?"
# docs = docsearch.similarity_search(query, include_metadata=True) # metadata is not necessary


# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "모니터를 PC에 연결할려면 어떻게 하면 되나요?"
query += "한국말로 설명해주세요."
# print(query)
docs = docsearch.similarity_search(query, include_metadata=True)

print(chain.run(input_documents=docs, question=query))
