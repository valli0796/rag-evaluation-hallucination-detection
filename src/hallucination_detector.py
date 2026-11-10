
import re


class HallucinationDetector:

    def detect(self, answer, retrieved_documents):
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        sentences = self._split_sentences(answer)

        if not retrieved_documents:
            unsupported = sentences
            supported = []
        else:
            context = " ".join(
                document["content"].lower()
                for document in retrieved_documents
            )

            supported = []
            unsupported = []

            for sentence in sentences:
                keywords = self._extract_keywords(sentence)

                if not keywords:
                    unsupported.append(sentence)
                    continue

                matches = sum(
                    1 for word in keywords if word in context
                )

                coverage = matches / len(keywords)

                if coverage >= 0.5:
                    supported.append(sentence)
                else:
                    unsupported.append(sentence)

        total = len(sentences)

        return {
            "supported_sentences": supported,
            "unsupported_sentences": unsupported,
            "total_sentences": total,
            "supported_count": len(supported),
            "unsupported_count": len(unsupported),
            "support_rate": round(
                len(supported) / total * 100, 2
            ) if total else 0.0
        }

    def _split_sentences(self, text):
        text = re.sub(r"(?m)^\s*\d+\.\s*", "", text)

        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

    def _extract_keywords(self, sentence):
        words = re.findall(
            r"\b[a-zA-Z]{4,}\b",
            sentence.lower()
        )

        stopwords = {
            "this", "that", "with", "from", "into",
            "when", "where", "which", "their", "there",
            "using", "works", "system", "answer"
        }

        return [
            word for word in words
            if word not in stopwords
        ]
