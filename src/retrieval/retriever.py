from typing import List, Dict, Any
import numpy as np
import torch
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Searching for query: {query}")
            
            # Правильно подготавливаем текстовые данные для ImageBind
            with torch.no_grad():
                # Используем функцию load_and_transform_text_data из imagebind.data
                text_input = data.load_and_transform_text([query], self.vector_store.model.device)
                inputs = {ModalityType.TEXT: text_input}
                
                # Получаем эмбеддинги через модель
                embeddings = self.vector_store.model(inputs)
                
                # Извлекаем текстовые эмбеддинги
                if ModalityType.TEXT in embeddings:
                    query_embedding = embeddings[ModalityType.TEXT]
                else:
                    logger.error("No text embeddings found in model output")
                    return []
            
            # Обрабатываем эмбеддинги
            query_embedding = self._process_embedding(query_embedding)
            
            if query_embedding is None:
                logger.error("Failed to process query embedding")
                return []
            
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            logger.debug(f"Query embedding type: {type(query_embedding)}")
            
            # Ищем похожие фрагменты
            results = self.vector_store.search(query_embedding, top_k)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.exception(f"Search failed: {str(e)}")
            return []
    
    def _process_embedding(self, embedding) -> np.ndarray:
        """Обрабатывает эмбеддинг в правильный формат"""
        try:
            # Конвертируем в numpy array
            if isinstance(embedding, torch.Tensor):
                embedding_np = embedding.cpu().numpy()
            elif isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], torch.Tensor):
                    embedding_np = embedding[0].cpu().numpy()
                else:
                    embedding_np = np.array(embedding)
            elif isinstance(embedding, np.ndarray):
                embedding_np = embedding
            else:
                logger.error(f"Unexpected embedding type: {type(embedding)}")
                return None
            
            # Обеспечиваем правильную размерность
            if embedding_np.ndim == 1:
                embedding_np = embedding_np.reshape(1, -1)
            elif embedding_np.ndim > 2:
                # Если больше 2 измерений, сжимаем до 2D
                embedding_np = embedding_np.reshape(embedding_np.shape[0], -1)
            
            # Приводим к float32
            embedding_np = embedding_np.astype(np.float32)
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to process embedding: {e}")
            return None