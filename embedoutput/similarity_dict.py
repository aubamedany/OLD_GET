from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models
from qdrant_client import QdrantClient
import torch
from scipy.spatial.distance import cosine
def minibatch(*tensors, **kwargs):

    batch_size = kwargs['batch_size']

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
def similarity_dict(name_data):
    name_outputs = f'/Users/namle/Desktop/GET/embed_probagation/output_{name_data}.pt'
    outputs = torch.load(name_outputs)
    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name=name_data,
        vectors_config=VectorParams(size=outputs.size(1), distance=Distance.COSINE),
    )
    id = range(outputs.size(0))
    for (minibatch_num,(idb,outputb)) in enumerate(minibatch(id,outputs,batch_size=32)):
        client.upsert(
            collection_name=name_data,
            points = models.Batch(
                ids = idb,
                vectors = outputb.tolist()
            )
        )
    similarity_dict = {}
    for i in range(outputs.size(0)):
        hits = client.search(
        collection_name=name_data,
        query_vector=outputs[i].tolist(),
        limit=10,
        )
        similarity_dict[i] = [hit.id for hit in hits]
    dict_name = f'embed_probagation/similar_dict_{name_data}.pt'
    torch.save(similarity_dict,dict_name)

def similar_main_snope():
    for i in range(5):
        name = f'snope_train_{i}'
        similarity_dict(name)
        name = f'snope_test_{i}'
        similarity_dict(name)
        if i ==4:
            name = f'snope_valid'
            similarity_dict(name)