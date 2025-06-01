import faiss
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path

class VectorStore:
    def __init__(self, model, dimension: int = 1024):
        self.model = model
        self.dimension = dimension
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.metadata_map = {}
        self.next_id = 0

    def add_video(self, video_path: str, metadata: Dict[str, Any], embeddings: np.ndarray) -> None:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Invalid embedding dimension: {embeddings.shape[1]}, expected {self.dimension}")
        
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        video_id = self.next_id
        self.index.add_with_ids(embeddings, np.array([video_id], dtype=np.int64))
        
        metadata['id'] = video_id
        self.metadata_map[video_id] = metadata
        self.next_id += 1

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] != -1:
                video_id = indices[0][i]
                results.append({
                    'metadata': self.metadata_map.get(video_id, {}),
                    'score': float(distances[0][i])
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix('.index')))
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump({
                'metadata_map': self.metadata_map,
                'next_id': self.next_id
            }, f)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path.with_suffix('.index')))
        with open(path.with_suffix('.json'), 'r') as f:
            data = json.load(f)
            self.metadata_map = data['metadata_map']
            self.next_id = data['next_id']