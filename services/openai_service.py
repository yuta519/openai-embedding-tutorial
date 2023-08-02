import os

from dotenv import load_dotenv
import openai
from openai.embeddings_utils import cosine_similarity

load_dotenv()


class OpenAiService:
    def __init__(self):
        self._embedding_model = "text-embedding-ada-002"
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def createVector(self, text: str) -> list[str]:
        response = openai.Embedding.create(model=self._embedding_model, input=text)
        result: list[str] = response["data"][0]["embedding"]
        return result

    def cosine_similarity(self, x: list[str], y: list[str]) -> int:
        return cosine_similarity(x, y)
