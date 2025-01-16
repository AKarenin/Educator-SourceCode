import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import api
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = api.OPENAPIKEY

filename = "/Users/akarenin/PycharmProjects/educator/venv/data/data.pdf"

loader = PyPDFLoader(filename)

pages = loader.load()
model = ChatOpenAI(model_name="gpt-4o", temperature=0)

text = ""
for page in pages:
    text += page.page_content

text = text.replace('\t', ' ')
num_tokens = model.get_num_tokens(text)


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2000, chunk_overlap=500)
docs = text_splitter.create_documents([text])
num_documents = len(docs)

print(docs[0])

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding=OpenAIEmbeddings(),
)
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
from langchain.chains import RetrievalQA


qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
while(1):
    question = input("Any questions about the textbook?")
    result = qa_chain({"query": question})
    print(result['result'])