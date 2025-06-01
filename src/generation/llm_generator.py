from typing import List, Dict, Any
import ollama
import logging

logger = logging.getLogger(__name__)

class LLMGenerator:
    '''
    Класс LLMGenerator генерирует ответы на основе найденного контекста с помощью языковой модели Ollama.
    '''
    def __init__(self, retriever, model_name: str = "qwen3:0.6b"):
        self.retriever = retriever
        self.model_name = model_name
        logger.info(f"Инициализация LLMGenerator с моделью: {model_name}")
        
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        # 1. Создает промпт из запроса и контекста найденных клипов
        # 2. Отправляет в модель Ollama (qwen3:0.6b)
        # 3. Возвращает сгенерированный ответ
        try:
            logger.info(f"Генерация ответа для запроса: '{query}'")
            prompt = self._create_prompt(query, context)
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.7}
            )
            
            return response['response']
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {str(e)}")
            return f"Ошибка генерации ответа: {str(e)}"
    
    def _create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"Клип {i+1}:\n"
            f"Время: {item['metadata']['start_time']:.1f}-{item['metadata']['end_time']:.1f}\n"
            f"Визуальное: {item['metadata']['visual_description']}"
            for i, item in enumerate(context)
        ])
        
        return f"""Проанализируйте содержание видео и ответьте на вопрос.
        
Контекст:
{context_text}

Вопрос: {query}

Ответ:"""