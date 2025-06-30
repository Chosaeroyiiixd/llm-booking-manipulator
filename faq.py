from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai

load_dotenv()

class HAUP_FAQ:

    def __init__(self, question):
        self.question = question
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.qdrant_client = QdrantClient(
                        url = os.getos("QDRANT_URL"), 
                        api_key = os.getenv("QDRANT_API_KEY"))
        self.collection_name = "haup-faq-collection"
        genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")
    
    def __prompting(self):
        question = self.question
        query_vector = self.embed_model.encode(question).tolist()
        
        search_results = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        limit=3,
                        query=query_vector,
                        with_payload=True
                    )
        
        similar_points = ""
        for result in search_results.points:
            similar_points += f"Q: {result.payload['question']}\nA: {result.payload['answer']}\n\n"

        system_prompt = """
        You are a smart assistant for the HAUP application. Your primary goal is to help users by answering their questions based on the provided context.

        Follow these two rules at all times:
        1.  Provide helpful, detailed answers based on the given information.
        2.  **Crucial:** You must respond in the exact same language as the user's question. For example, if the user asks in Thai, your answer must be entirely in Thai.

        """

        gemini_prompt = f"""
        {system_prompt}
        {similar_points}
        User Question : {question}
        """
        return {"gemini_prompt" : gemini_prompt, "search_results" : search_results.points}
    
    def gemini_response(self):
        gemini_prompt = self.__prompting()
        
        if len(gemini_prompt['search_results']) > 0:
            response = self.gemini.generate_content(gemini_prompt["gemini_prompt"]).text
            print(response)
        else:
            print("Contact customer service.")

if __name__ == '__main__':
    print(HAUP_FAQ('can I return a car others place?').gemini_response())