import pandas as pd
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import requests
import hashlib
import uuid

load_dotenv()

class QdrantUpserter:
    def __init__(self):
        self.haup_faq_url = f"{os.getenv('GGSHEET_URL')}+{os.getenv('GGSHEET_API_KEY')}"
        self.qdrant_client = QdrantClient(
                    url = os.getenv("QDRANT_URL"), 
                    api_key = os.getenv("QDRANT_API_KEY"))
        self.collection_name = "haup-faq-collection"
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def create_collection(self):
        collections = self.qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if self.collection_name not in collection_names:
            print(f'no collection : "{self.collection_name}", creating.. ')
            self.qdrant_client.create_collection(collection_name = self.collection_name, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            print(f'collection : "{self.collection_name}" created!')
        else:
            pass

    def get_ggsheets_data(self):
        print('getting ggsheets FAQ data..')
        faq_data = requests.get(self.haup_faq_url).json()

        qna_df = pd.DataFrame(faq_data['values'], columns=['question', 'answer'])
        qna_df['content'] = qna_df['question'] + ' ' + qna_df['answer']
        qna_df.insert(loc=0, column='point_id', value = qna_df['content'].apply(lambda row :hashlib.md5(row.encode()).hexdigest()))

        return qna_df
    
    def embed_fn(self, data):
        emded_model = self.embed_model
        embeddings = emded_model.encode(data).tolist()
        return embeddings
    
    def check_duplicate(self):
        data = self.get_ggsheets_data()
        all_QnA = list(data["point_id"])
        qdrant_pts = self.qdrant_client.retrieve(collection_name=self.collection_name, ids=all_QnA)
        duplicate_pts = [pts for pts in [qdrant_pts[i].id for i in range(len(qdrant_pts))]]

        new_qna_df = data[~(data['point_id'].apply(lambda x : str(uuid.UUID(x)))).isin(duplicate_pts)]
        print(f'data to upsert : {len(new_qna_df)} point(s).')
        return new_qna_df

    def upsert_qdrant(self):
        self.create_collection()
        upsert_data = self.check_duplicate()
        if len(upsert_data) > 0:
            embeddings = self.embed_fn(upsert_data["content"].tolist())
            points = [
                PointStruct(
                    id=str(uuid.UUID(upsert_data.iloc[i]["point_id"])),
                    vector=embeddings[i],
                    payload={
                        "question": upsert_data.iloc[i]["question"],
                        "answer": upsert_data.iloc[i]["answer"]
                    }
                )
                for i in range(len(upsert_data))
            ]

            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
            return print(f"💾 Pushed new {len(upsert_data)} Q&A(s) to Qdrant.")
        else:
            return print('✅ No new data to pushed to Qdrant Cloud.')
        
if __name__ == '__main__':
    QdrantUpserter().upsert_qdrant()