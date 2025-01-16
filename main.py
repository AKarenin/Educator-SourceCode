import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

import api

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = api.OPENAPIKEY


filename = "/Users/akarenin/PycharmProjects/educator/venv/data/data.pdf"

loader = PyPDFLoader(filename)
pages = loader.load()

# Concatenate and preprocess text
text = ""
for page in pages:
    text += page.page_content
text = text.replace('\t', ' ')

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2000, chunk_overlap=500)
docs = text_splitter.create_documents([text])

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
)

# Load RAG Prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize Chat Model and RetrievalQA
model = ChatOpenAI(model_name="gpt-4", temperature=0)
retriever = vectorstore.as_retriever(search_type="similarity")
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Feedback log file
feedback_file = "feedback_log.json"

# Load or initialize feedback log
if os.path.exists(feedback_file):
    with open(feedback_file, "r") as f:
        feedback_log = json.load(f)
else:
    feedback_log = []

# Helper function to prioritize manual corrections
def get_answer_with_feedback(question: str):
    """
    Check for a manual correction in the feedback log first.
    If found, return it. Otherwise, use the QA chain.
    """
    for feedback in feedback_log:
        if feedback["question"].lower() == question.lower():
            print("Manual Correction Found!")
            return feedback["correct_response"]
    result = qa_chain({"query": question})
    return result['result']

# Main loop for Q&A and feedback
while True:
    question = input("Any questions about the textbook? (type 'exit' to quit): ").strip()
    if question.lower() == "exit":
        break

    # Fetch the response
    response = get_answer_with_feedback(question)
    print("GPT's Response:", response)

    # Handle user feedback
    user_input = input("Are you satisfied with this answer? Type 'yes', 'no', or 'still stuck': ").strip().lower()

    if user_input == "still stuck":
        # User provides the correct answer
        manual_response = input("Provide the correct answer: ").strip()

        # Save the correction in the feedback log
        feedback_log.append({
            "question": question,
            "incorrect_response": response,
            "correct_response": manual_response
        })

        print("Your response has been saved and will be used for future improvement.")

        # Add the manual correction to the vector store
        corrected_doc = Document(page_content=manual_response, metadata={"source": "manual_feedback"})
        vectorstore.add_documents([corrected_doc])

        # Reinitialize and refresh retriever
        retriever = vectorstore.as_retriever(search_type="similarity")
        qa_chain.retriever = retriever

        print("Manual correction added to vectorstore and retriever refreshed.")
    elif user_input == "no":
        print("Thank you for your feedback. We'll use this to improve.")
    else:
        print("Glad the response was helpful!")

# Save feedback log to file
with open(feedback_file, "w") as f:
    json.dump(feedback_log, f, indent=4)
