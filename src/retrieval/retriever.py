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
            logger.info(f"Выполнение поиска по запросу: {query}")
            
            # Подготовка текстовых данных для ImageBind
            with torch.no_grad():
                text_input = data.load_and_transform_text([query], self.vector_store.model.device)
                inputs = {ModalityType.TEXT: text_input}
                
                embeddings = self.vector_store.model(inputs)
                
                if ModalityType.TEXT in embeddings:
                    query_embedding = embeddings[ModalityType.TEXT]
                else:
                    logger.error("Текстовые эмбеддинги не найдены в выходных данных модели")
                    return []
            
            query_embedding = self._process_embedding(query_embedding)
            
            if query_embedding is None:
                logger.error("Ошибка обработки эмбеддинга запроса")
                return []
            
            logger.debug(f"Форма эмбеддинга запроса: {query_embedding.shape}")
            logger.debug(f"Тип эмбеддинга запроса: {type(query_embedding)}")
            
            results = self.vector_store.search(query_embedding, top_k)
            
            logger.info(f"Найдено результатов: {len(results)}")
            return results
            
        except Exception as e:
            logger.exception(f"Ошибка поиска: {str(e)}")
            return []
    
    def _process_embedding(self, embedding) -> np.ndarray:
        try:
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
                logger.error(f"Неподдерживаемый тип эмбеддинга: {type(embedding)}")
                return None
            
            if embedding_np.ndim == 1:
                embedding_np = embedding_np.reshape(1, -1)
            elif embedding_np.ndim > 2:
                embedding_np = embedding_np.reshape(embedding_np.shape[0], -1)
            
            embedding_np = embedding_np.astype(np.float32)
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Ошибка обработки эмбеддинга: {e}")
            return None