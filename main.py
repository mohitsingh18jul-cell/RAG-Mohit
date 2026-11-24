
import os
import sys
import warnings
import argparse

warnings.filterwarnings("ignore")

# The Windows console commonly uses cp1252. Keep PDF/LLM text from crashing
# the application when it contains characters unsupported by that encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

provider = "nvidia" if os.getenv("NVIDIA_API_KEY") else "groq"
api_key = os.getenv("NVIDIA_API_KEY") if provider == "nvidia" else os.getenv("GROQ_API_KEY")

# Accept an existing key-only .env file while supporting the standard named format.
if not api_key and os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if line.startswith("NVIDIA_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"\'')
                break
            if line.startswith("GROQ_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"\'')
                break
            if line and "=" not in line:
                api_key = line

if not api_key:
    raise ValueError(
        "No API key found. Add GROQ_API_KEY or NVIDIA_API_KEY to .env"
    )


# 2. Document Loading
pdf_candidates = (
    os.path.join(BASE_DIR, "DA sample paper 2026.pdf"),
    os.path.join(BASE_DIR, "document_loader", "DA sample paper 2026.pdf"),
)
pdf_path = next(
    (path for path in pdf_candidates if os.path.isfile(path)),
    None,
)

if pdf_path is None:
    raise FileNotFoundError(
        "DA sample paper 2026.pdf"
        "document_loader/DA sample paper 2026.pdf."
    )

print(f"Loading document from: {pdf_path}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} document pages.")


# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} text chunks.")


# 4. Embeddings and Qdrant Vector Store
print("Creating embeddings and indexing into Qdrant...")

embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings
)

vectorstore.add_documents(chunks)

print("Indexing complete!")


# 5. Initialize the configured LLM provider
if provider == "groq":
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=2500
    )
else:
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500
    )


# 6. Retriever and RAG Prompt
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

system_prompt = """
You are an expert GATE Data Science and Artificial Intelligence
(DA) Exam Preparation Tutor.

You have access to a PDF document named
'DA sample paper 2026.pdf'.

YOUR MAIN RESPONSIBILITY:
Help students understand, solve, and practice questions
from the uploaded GATE DA sample paper.

INSTRUCTIONS:

1. ANSWER USING PDF CONTEXT
- Use the retrieved PDF context as your primary source.
- Answer questions based on the provided context.
- Do not invent questions, options, or information.
- If the required information is not available in the
  retrieved context, clearly say so.
- You may use your mathematical and programming knowledge
  to explain or solve a question, but distinguish your
  explanation from information explicitly stated in the PDF.

2. QUESTION SOLVING
When a student asks a question:
- Identify the question number if available.
- Explain what the question is asking.
- Identify the relevant concept or formula.
- Solve the problem step by step.
- Explain the calculation and logic.
- Provide the final answer clearly.
- For MCQs, explain why the correct option is correct.
- If useful, briefly explain why other options are incorrect.

3. SUBJECT EXPLANATION
Help students understand topics related to:
- Artificial Intelligence
- Machine Learning
- Data Science
- Probability and Statistics
- Linear Algebra
- Calculus
- Database Management Systems
- Data Structures and Algorithms
- Programming and Python
- General Aptitude

4. TEACHING STYLE
- Use simple English.
- Assume the student is a beginner.
- Avoid unnecessarily complicated explanations.
- Give practical examples wherever useful.
- Explain mathematical formulas step by step.
- Explain Python code line by line when requested.

5. EXAM PREPARATION
When asked:
- Generate practice questions based on the PDF topics.
- Create MCQs and numerical problems.
- Conduct mock tests.
- Explain difficult questions.
- Provide important formulas and short revision notes.
- Help students prepare for GATE DA.

6. RESPONSE FORMAT

For numerical or MCQ questions, use:

Question:
[Question statement]

Concept:
[Relevant concept]

Formula:
[Required formula, if applicable]

Step-by-step Solution:
[Detailed explanation]

Final Answer:
[Correct answer and option, if applicable]

7. IMPORTANT RULES
- Never pretend that an answer is present in the PDF
  if it is not available in the retrieved context.
- Never provide an unsupported official answer key.
- If a question contains a diagram, use the retrieved
  visual information when available.
- Mention the PDF page number or question number
  whenever it is available.
- For multiple-correct-answer questions, evaluate
  every option separately.

Your goal is to act as a personal GATE DA tutor,
helping students learn concepts, solve questions,
and prepare effectively for the examination.

Retrieved Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

rag_chain = prompt | llm | StrOutputParser()


# 7. RAG Query Function
def query_rag(question: str):

    print("\n" + "=" * 60)
    print(f"Query: {question}")
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)

    print(f"\nRetrieved {len(retrieved_docs)} relevant chunks:")

    for i, doc in enumerate(retrieved_docs, start=1):

        page_num = doc.metadata.get("page", "Unknown")

        preview = (
            doc.page_content
            .strip()
            .replace("\n", " ")[:150]
        )

        print(f"[{i}] Page {page_num}: {preview}...")

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    print(f"\nGenerating response from {provider.upper()} LLM...")

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:")
    print(response)

    print("=" * 60)

    return response


# 8. Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask questions from DA sample paper 2026.pdf"
    )
    parser.add_argument(
        "-q",
        "--question",
        default=(
            "Solve Q.11 from the PDF. For a classification problem, PCA "
            "was used to reduce the feature space from 100 dimensions to "
            "10 dimensions. What is the angle theta between the first and "
            "tenth principal components?"
        ),
        help="Question to answer using the PDF context",
    )
    args = parser.parse_args()

    query_rag(args.question)
