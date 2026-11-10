
import asyncio
import json
from openai import AsyncOpenAI
from src.rag_pipeline import RAGPipeline


async def main():
    question = "How does a RAG system work?"

    pipeline = RAGPipeline()
    result = pipeline.ask(question)
    answer = result["answer"]

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    prompt = f"""
Extract the factual statements from the answer below.

Return ONLY valid JSON in this exact format:
{{"statements": ["statement 1", "statement 2"]}}

Do not include explanations, schemas, markdown, or any other text.

Answer:
{answer}
"""

    print("Testing simplified JSON extraction...")

    try:
        response = await client.chat.completions.create(
            model="llama3.2:3b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content

        print("\nRAW RESPONSE:")
        print(raw_output)

        parsed = json.loads(raw_output)

        print("\nEXTRACTED STATEMENTS:")
        for statement in parsed["statements"]:
            print("-", statement)

    except Exception as error:
        print("\nERROR:", type(error).__name__, str(error))

    finally:
        await client.close()


asyncio.run(main())
