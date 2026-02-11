from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import requests
import os
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


class TLS12HttpAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=ctx,
            **pool_kwargs,
        )


def make_session_tls12() -> requests.Session:
    s = requests.Session()
    s.mount("https://", TLS12HttpAdapter())
    return s


client = QdrantClient(url="http://localhost:6333")
COLLECTION = "cw01_collection"
filename = ["data_01.txt", "data_02.txt", "data_03.txt", "data_04.txt", "data_05.txt"]
texts = []
PATH = "HW/Day5/HW/"


for file in filename:
    path = os.path.join(PATH, file)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        texts.append(content)

session = make_session_tls12()
all_vectors = []
dimension = None
for start in range(0, len(texts), 1):
    chunk_text = texts[start : start + 1]

    data = {"texts": chunk_text, "normalize": True, "batch_size": 32}
    try:
        resp = session.post("https://ws-04.wade0426.me/embed", json=data, timeout=60)
        print("Server OK!")
    except Exception as e:
        print(e)
    resp.raise_for_status()
    result = resp.json()

    if dimension is None:
        dimension = result["dimension"]
    elif dimension != result["dimension"]:
        raise RuntimeError(f"Embedding 維度不同{dimension} vs {data['dimension']}")
    all_vectors.extend(result["embeddings"])

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

points = []
for i, (text, vector, fname) in enumerate(zip(texts, all_vectors, filename), start=1):
    points.append(
        PointStruct(id=i, vector=vector, payload={"text": text, "source": fname})
    )

client.upsert(
    collection_name=COLLECTION,
    points=points,
    wait=True,
)

print("END")
