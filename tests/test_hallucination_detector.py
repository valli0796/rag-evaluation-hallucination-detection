
from src.rag_pipeline import RAGPipeline
from src.hallucination_detector import HallucinationDetector

rag = RAGPipeline()
detector = HallucinationDetector()

question = "How does a RAG system work?"

result = rag.ask(question)

evaluation = detector.detect(
    answer=result["answer"],
    retrieved_documents=result["retrieved_documents"]
)

print("\nHallucination Detection Results")
print("Total sentences:", evaluation["total_sentences"])
print("Supported:", evaluation["supported_count"])
print("Unsupported:", evaluation["unsupported_count"])
print("Keyword support rate:", evaluation["support_rate"], "%")

print("\nUnsupported Sentences:")

for sentence in evaluation["unsupported_sentences"]:
    print("-", sentence)
