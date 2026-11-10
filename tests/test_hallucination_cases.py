
from src.evaluator import RAGEvaluator


def main():
    evaluator = RAGEvaluator()

    retrieved_documents = [
        {
            "content": (
                "Retrieval-Augmented Generation combines information "
                "retrieval with large language models. "
                "RAG can use relevant external documents when answering "
                "questions. "
                "RAG can reduce unsupported answers but cannot guarantee "
                "100% answer accuracy."
            ),
            "metadata": {"source": "sample.pdf", "page": 1},
            "distance": 0.5
        }
    ]

    test_cases = [
        {
            "name": "Supported claim",
            "statement": (
                "RAG combines information retrieval with language models."
            ),
            "expected": True
        },
        {
            "name": "Unsupported claim",
            "statement": (
                "RAG systems use blockchain technology for storage."
            ),
            "expected": False
        },
        {
            "name": "Paraphrased claim",
            "statement": (
                "RAG uses external documents to answer user questions."
            ),
            "expected": True
        },
        {
            "name": "Contradictory claim",
            "statement": (
                "RAG cannot use external documents."
            ),
            "expected": False
        }
    ]

    results = evaluator.verify_statements(
        statements=[
            case["statement"] for case in test_cases
        ],
        retrieved_documents=retrieved_documents
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print("\nHALLUCINATION DETECTION TEST SUITE")
    print("=" * 55)

    for index, (case, result) in enumerate(
        zip(test_cases, results), start=1
    ):
        expected = case["expected"]
        actual = result["supported"]

        if expected and actual:
            tp += 1
        elif not expected and not actual:
            tn += 1
        elif not expected and actual:
            fp += 1
        else:
            fn += 1

        print(f"\nTest {index}: {case['name']}")
        print("Claim:", case["statement"])
        print("Expected:", "SUPPORTED" if expected else "UNSUPPORTED")
        print("Actual:", "SUPPORTED" if actual else "UNSUPPORTED")
        print("Keyword coverage:", result["keyword_coverage"], "%")
        print(
            "Contradiction detected:",
            result["contradiction_detected"]
        )
        print("Result:", "PASS" if expected == actual else "FAIL")

    total = len(test_cases)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    print("\nCONFUSION MATRIX")
    print("=" * 55)
    print("True Positives (TP):", tp)
    print("True Negatives (TN):", tn)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)

    print("\nEVALUATION METRICS")
    print("=" * 55)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-score:", round(f1 * 100, 2), "%")

    passed = sum(
        case["expected"] == result["supported"]
        for case, result in zip(test_cases, results)
    )

    print("\nFINAL TEST SUMMARY")
    print("=" * 55)
    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", total - passed)
    print("Test pass rate:", round(passed / total * 100, 2), "%")


if __name__ == "__main__":
    main()
