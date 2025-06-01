from typing import List, Dict, Any
import ollama
import logging

logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, retriever, model_name: str = "qwen3:0.6b"):
        self.retriever = retriever
        self.model_name = model_name
        logger.info(f"Initialized LLMGenerator with model: {model_name}")
        
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        try:
            logger.info(f"Generating response for query: '{query}'")
            prompt = self._create_prompt(query, context)
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.7}
            )
            
            return response['response']
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"Clip {i+1}:\n"
            f"Time: {item['metadata']['start_time']:.1f}-{item['metadata']['end_time']:.1f}\n"
            f"Visual: {item['metadata']['visual_description']}"
            for i, item in enumerate(context)
        ])
        
        return f"""Analyze video content and answer the question.
        
Context:
{context_text}

Question: {query}

Answer:"""