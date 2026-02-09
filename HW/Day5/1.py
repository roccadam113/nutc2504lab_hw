import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def Ppoints(client, id: int, payload: dict, collecttion_name: str):
    client.upsert(
        collecttion_name=collecttion_name,
        points=[PointStruct(id=id, vector=vector, payload=payload)],
    )


client = QdrantClient(url="https://localhost:6333")
