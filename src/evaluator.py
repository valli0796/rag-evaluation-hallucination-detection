
import re


class RAGEvaluator:
    def __init__(self):
        self.stopwords = {
            "this", "that", "with", "from", "into",
            "when", "where", "which", "their", "there",
            "using", "works", "system", "systems",
            "answer", "the", "and", "for", "are",
            "was", "were", "has", "have", "can",
            "its", "a", "an", "to", "of", "in",
            "on", "by", "is", "be", "as", "or",
            "always", "cannot"
        }

        self.negation_words = {
            "not", "never", "no", "cannot",
            "can't", "don't", "doesn't",
            "isn't", "aren't", "won't",
            "without", "unable"
        }

    def extract_statements(self, answer):
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        statements = []

        for line in answer.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(
                r"^\s*(?:\d+[\.\)]|[-*•])\s*",
                "",
                line
            )

            if line.endswith(":"):
                continue

            parts = re.split(r"(?<=[.!?])\s+", line)

            for part in parts:
                statement = part.strip()

                if statement and len(statement.split()) >= 3:
                    if statement not in statements:
                        statements.append(statement)

        if not statements:
            raise ValueError("No factual statements were extracted.")

        return statements

    def _keywords(self, text):
        words = re.findall(
            r"\b[a-zA-Z']{3,}\b",
            text.lower()
        )

        return {
            word for word in words
            if word not in self.stopwords
        }

    def _has_negation(self, text):
        words = set(
            re.findall(r"\b[a-zA-Z']+\b", text.lower())
        )

        return bool(words.intersection(self.negation_words))

    def verify_statements(self, statements, retrieved_documents):
        context = " ".join(
            document["content"]
            for document in retrieved_documents
        )

        evidence_sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+",
                context
            )
            if sentence.strip()
        ]

        results = []

        for statement in statements:
            statement_words = self._keywords(statement)

            best_score = 0
            best_evidence = ""

            for evidence in evidence_sentences:
                evidence_words = self._keywords(evidence)

                if not statement_words:
                    continue

                overlap = statement_words.intersection(
                    evidence_words
                )

                score = len(overlap) / len(statement_words)

                if score > best_score:
                    best_score = score
                    best_evidence = evidence

            contradiction = False

            if best_evidence and best_score >= 0.60:
                statement_negative = self._has_negation(statement)
                evidence_negative = self._has_negation(best_evidence)

                if statement_negative != evidence_negative:
                    contradiction = True

            supported = (
                best_score >= 0.60
                and not contradiction
            )

            results.append({
                "statement": statement,
                "supported": supported,
                "evidence": best_evidence if supported else "",
                "keyword_coverage": round(best_score * 100, 2),
                "contradiction_detected": contradiction
            })

        return results

    def evaluate_faithfulness(
        self,
        question,
        answer,
        retrieved_documents
    ):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not retrieved_documents:
            raise ValueError("Retrieved documents cannot be empty.")

        statements = self.extract_statements(answer)

        results = self.verify_statements(
            statements,
            retrieved_documents
        )

        supported_count = sum(
            item["supported"] for item in results
        )

        total_count = len(results)
        unsupported_count = total_count - supported_count

        score = (
            supported_count / total_count
            if total_count else 0
        )

        return {
            "question": question,
            "total_statements": total_count,
            "supported_count": supported_count,
            "unsupported_count": unsupported_count,
            "faithfulness_score": round(score, 4),
            "faithfulness_percentage": round(score * 100, 2),
            "evaluation_method": (
                "Keyword matching with basic negation detection"
            ),
            "statement_results": results
        }
