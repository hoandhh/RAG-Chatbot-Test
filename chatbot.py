import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (e.g., API keys) from .env file
load_dotenv()

# Create a basic text splitter for initial setup (can be used for experimentation)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Load all PDF documents from the ./papers directory using UnstructuredFileLoader for parsing
loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=False,
)

# Actually load the documents from disk
docs = loader.load()

# Define separators for splitting PDFs and markdown, tuned for scientific papers and notes
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# Refine the text splitter with better chunking and separators, useful for markdown-heavy or structured PDFs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

# Split all loaded documents into smaller, manageable text chunks
splits = text_splitter.split_documents(docs)

# Initialize Gemini (Google Generative AI) embeddings for semantic vectorization of chunks
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# Create a FAISS vector store for fast similarity search on chunk embeddings
vectorstore = FAISS.from_documents(
    documents=splits, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE
)

# Set up a retriever for semantic search from the vector store, with score threshold to avoid irrelevant info
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2},
)

# Create the prompt template for strict, source-citing QA on your internal documents
template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# Format the prompt using LangChain's chat template
prompt = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# Build a retrieval-augmented generation (RAG) pipeline:
# 1. Retrieve relevant chunks
# 2. Construct the prompt
# 3. Generate an answer
# 4. Parse the output as a string
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example question to test the whole pipeline
question = "Consult the document to find out the main idea of the paper and list the main points."
response = rag_chain.invoke(question)

# Output the model's response to the console
pprint(response)