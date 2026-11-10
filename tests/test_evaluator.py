
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator


def main():
    print("Initializing RAG pipeline and evaluator...")

    pipeline = RAGPipeline()
    evaluator = RAGEvaluator()

    question = "How does a RAG system work?"

    print("\nGenerating RAG answer...")
    result = pipeline.ask(question)

    print("\nGENERATED ANSWER")
    print("=" * 50)
    print(result["answer"])

    # Add a deliberately unsupported claim
    test_answer = (
        result["answer"]
        + "\nRAG systems always guarantee 100% answer accuracy."
    )

    print("\nRunning evidence-based evaluation...")

    evaluation = evaluator.evaluate_faithfulness(
        question=question,
        answer=test_answer,
        retrieved_documents=result["retrieved_documents"]
    )

    print("\nEVALUATION RESULTS")
    print("=" * 50)

    print("Evaluation method:", evaluation["evaluation_method"])
    print("Total statements:", evaluation["total_statements"])
    print("Supported statements:", evaluation["supported_count"])
    print("Unsupported statements:", evaluation["unsupported_count"])
    print(
        "Evidence support score:",
        evaluation["faithfulness_percentage"],
        "%"
    )

    print("\nSTATEMENT-WISE EVALUATION")
    print("=" * 50)

    for index, item in enumerate(
        evaluation["statement_results"], start=1
    ):
        print(f"\nStatement {index}:")
        print(item["statement"])

        status = "SUPPORTED" if item["supported"] else "UNSUPPORTED"
        print("Status:", status)
        print("Keyword coverage:", item["keyword_coverage"], "%")

        if item["evidence"]:
            print("Evidence:", item["evidence"])
        else:
            print("Evidence: No matching evidence found.")

    print("\nTEST VALIDATION")
    print("=" * 50)

    unsupported_claim_detected = any(
        not item["supported"]
        and "100% answer accuracy" in item["statement"].lower()
        for item in evaluation["statement_results"]
    )

    if unsupported_claim_detected:
        print("PASS: Deliberately unsupported claim was detected.")
    else:
        print("FAIL: Unsupported claim was not detected.")

    print("\nEvaluation test completed.")


if __name__ == "__main__":
    main()
