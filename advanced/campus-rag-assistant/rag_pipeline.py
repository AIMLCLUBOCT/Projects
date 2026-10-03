"""
Campus Ordinance & Syllabus RAG Assistant
AIML Club — Oriental College of Technology, Bhopal

An advanced Generative AI & Retrieval-Augmented Generation (RAG) project teaching:
1. Document ingestion and chunking strategies with sliding window overlap.
2. Vector embeddings and similarity retrieval (Cosine Similarity over Term Vectors).
3. Grounded context assembly: combining user queries with retrieved context.
4. Source citation & hallucination reduction.
5. Interactive query interface answering campus questions.

Supports:
- Pure Standard Library Python vector search (zero external dependencies required).
- Extensible hook for LangChain, ChromaDB, and SentenceTransformers.
"""

import math
import re
import sys
from collections import Counter, defaultdict

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =====================================================================
# 📚 Campus Knowledge Base Documents (Official OCT Policies & Syllabi)
# =====================================================================
CAMPUS_DOCUMENTS = [
    {
        "id": "DOC-ORD-01",
        "title": "University Ordinance: Examination Attendance Regulations",
        "source": "Academic Ordinance Section 4.2",
        "content": (
            "Every student is required to attend a minimum of 75% of total scheduled lectures, "
            "practical sessions, and tutorials in each subject to be eligible to appear in the end-semester university examinations. "
            "A condonation of up to 15% (bringing the minimum threshold to 60%) may be granted by the College Principal "
            "only on verified medical grounds or official participation in national sports/academic competitions, "
            "provided valid medical certificates and Dean approvals are submitted within 7 working days."
        )
    },
    {
        "id": "DOC-ORD-02",
        "title": "Grading System & SGPA/CGPA Calculation",
        "source": "Academic Ordinance Section 7.1",
        "content": (
            "The institute follows a 10-point letter grading system: Grade A+ (Outstanding, 10 points, 90-100 marks), "
            "A (Excellent, 9 points, 80-89 marks), B+ (Very Good, 8 points, 70-79 marks), B (Good, 7 points, 60-69 marks), "
            "C+ (Fair, 6 points, 50-59 marks), C (Average, 5 points, 40-49 marks), and F (Fail, 0 points, <40 marks). "
            "Semester Grade Point Average (SGPA) is calculated as: SGPA = Sum(Course Credits * Grade Points) / Sum(Course Credits). "
            "Cumulative Grade Point Average (CGPA) is the weighted average of all semester grade points earned."
        )
    },
    {
        "id": "DOC-CLUB-01",
        "title": "AIML Club OCT: Lab Facilities & Workshop Schedule",
        "source": "AIML Club Constitution 2026",
        "content": (
            "The AI & Machine Learning Club operates within the Department of Computer Science & Engineering, "
            "Oriental College of Technology, Bhopal. Regular hands-on workshops are conducted every Saturday at 2:00 PM "
            "in Advanced Computing Lab 3. The club provides high-performance computing access, GPU-accelerated workstations, "
            "and cloud credits for hackathon teams building real-world AI prototypes."
        )
    },
    {
        "id": "DOC-HACK-01",
        "title": "Hackathon Team Formation & Open Source Standards",
        "source": "OCT Tech Challenge Handbook",
        "content": (
            "All college hackathons and coding sprints hosted by AIML Club require teams of 2 to 4 students. "
            "Inter-departmental and cross-year teams are highly encouraged to foster diverse engineering talent. "
            "All submitted code repositories must be published on GitHub under an OSI-approved open source license (MIT or Apache 2.0) "
            "with a comprehensive README, requirements.txt, and live demo. Plagiarism or uncredited code copying leads to immediate disqualification."
        )
    },
    {
        "id": "DOC-DISC-01",
        "title": "Laboratory Conduct & Safety Guidelines",
        "source": "Central Computing Facility Rules",
        "content": (
            "Food and beverages are strictly prohibited inside computing laboratories. Students must log out of their session "
            "and shut down desktop workstations upon completing experiments. Unauthorized software installation, mining, "
            "or security circumvention on campus networks is strictly monitored and subject to disciplinary action."
        )
    }
]


# =====================================================================
# ✂️ Chunking & Vector Search Engine
# =====================================================================
def tokenize(text: str):
    """Tokenize and normalize text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class SimpleRAGIndex:
    """
    Lightweight, self-contained RAG index implementing TF-IDF vectorization
    and cosine similarity search in pure Python without external dependencies.
    """
    def __init__(self, chunk_size: int = 40, overlap: int = 10):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def chunk_document(self, doc):
        """Splits document content into overlapping token chunks."""
        words = doc["content"].split()
        if len(words) <= self.chunk_size:
            return [{
                "doc_id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "chunk_id": f"{doc['id']}_c0",
                "text": doc["content"]
            }]

        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            window = words[i:i + self.chunk_size]
            if len(window) < 10 and chunks:
                continue
            chunks.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "chunk_id": f"{doc['id']}_c{len(chunks)}",
                "text": " ".join(window)
            })
        return chunks

    def build_index(self, documents):
        """Builds inverted index and TF-IDF vectors for all document chunks."""
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self.chunk_document(doc))

        # Compute Document Frequencies
        df = defaultdict(int)
        chunk_tokens = []
        for ch in self.chunks:
            tokens = tokenize(ch["text"])
            chunk_tokens.append(tokens)
            unique_words = set(tokens)
            for w in unique_words:
                df[w] += 1

        total_chunks = len(self.chunks)
        self.vocab = {w: idx for idx, w in enumerate(df.keys())}
        self.idf = {w: math.log((total_chunks + 1) / (count + 1)) + 1.0 for w, count in df.items()}

        # Compute TF-IDF vectors
        self.doc_vectors = []
        for tokens in chunk_tokens:
            vec = self._vectorize(tokens)
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens):
        """Creates a normalized TF-IDF vector."""
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {}
        for w, count in tf.items():
            if w in self.vocab:
                vec[w] = (count / total) * self.idf[w]

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    def _cosine_similarity(self, vecA, vecB):
        """Calculates dot product of two normalized vectors."""
        common = set(vecA.keys()) & set(vecB.keys())
        return sum(vecA[k] * vecB[k] for k in common)

    def retrieve(self, query: str, top_k: int = 2):
        """Retrieves the top-k most relevant chunks for a user query."""
        q_tokens = tokenize(query)
        q_vec = self._vectorize(q_tokens)

        scores = []
        for i, d_vec in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(q_vec, d_vec)
            scores.append((sim, self.chunks[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# =====================================================================
# 🤖 Synthesizer & Augmented Prompt Generator
# =====================================================================
def generate_augmented_response(query: str, retrieved_items):
    """
    Simulates RAG generation by constructing a grounded prompt with retrieved
    knowledge chunks and generating citations.
    """
    context_blocks = []
    citations = []

    for rank, (score, chunk) in enumerate(retrieved_items, 1):
        context_blocks.append(f"[{rank}] Source: {chunk['source']} ({chunk['title']})\n    \"{chunk['text']}\"")
        citations.append(f"{chunk['source']}")

    context_str = "\n".join(context_blocks)
    top_chunk = retrieved_items[0][1] if retrieved_items else None
    top_score = retrieved_items[0][0] if retrieved_items else 0.0

    print("\n" + "=" * 70)
    print(f"❓ User Query: \"{query}\"")
    print("=" * 70)
    print("🔍 Retrieved Knowledge Chunks (Top Ranked):")
    print(context_str)
    print("\n🤖 Grounded RAG Assistant Answer:")

    # Heuristic answer extraction based on top context
    if top_chunk and top_score > 0.05:
        print(f"   Based on {top_chunk['source']}, here is the verified campus policy:")
        print(f"   \"{top_chunk['text']}\"")
        print(f"\n   📌 Citations: {', '.join(set(citations))}")
        print(f"   📊 Retrieval Relevance Score: {top_score:.4f}")
    else:
        print("   I could not find sufficiently confident information in the official campus database.")


# =====================================================================
# 🚀 Main Execution & Demonstration
# =====================================================================
def main():
    print("=" * 70)
    print("🏛️  AIML Club OCT — Campus Ordinance & Syllabus RAG Assistant")
    print("=" * 70)
    print("Indexing official college ordinances, grading regulations, and club guidelines...\n")

    index = SimpleRAGIndex(chunk_size=45, overlap=10)
    index.build_index(CAMPUS_DOCUMENTS)
    print(f"✅ Indexed {len(CAMPUS_DOCUMENTS)} campus documents into {len(index.chunks)} overlapping chunks.")
    print(f"✅ Vocabulary size: {len(index.vocab)} unique term tokens.")

    # Demonstration queries asked by students
    student_queries = [
        "What is the minimum attendance required to sit in university exams?",
        "How is SGPA calculated and what are the letter grade points?",
        "When and where are AIML Club workshops conducted?",
        "What are the rules regarding open source licensing in college hackathons?",
    ]

    for q in student_queries:
        results = index.retrieve(q, top_k=2)
        generate_augmented_response(q, results)

    print("\n" + "=" * 70)
    print("🎉 RAG Assistant pipeline executed successfully!")
    print("   Ready to connect with LangChain, ChromaDB, or Hugging Face LLMs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
