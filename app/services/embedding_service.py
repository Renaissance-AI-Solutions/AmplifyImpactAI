from sentence_transformers import SentenceTransformer as STModel
from typing import List, Union, Optional # Added Optional
import numpy as np
import logging
from flask import current_app # For config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = None, normalize: bool = True):
        self.model_name = model_name or current_app.config.get('KB_EMBEDDING_MODEL_NAME', 'BAAI/bge-large-en-v1.5')
        self.normalize_embeddings = normalize
        self.client: Optional[STModel] = None
        self._dimension: int = 0
        # Embedding Caching - Suggested Enhancement
        self.enable_cache = current_app.config.get('KB_EMBEDDING_ENABLE_CACHE', False)
        self.cache: Optional[dict] = {} if self.enable_cache else None

        self._initialize_client()

        # Define prefixes based on model type - BGE and E5 have similar needs
        # These could also be moved to config if they vary significantly by model.
        if "bge-" in self.model_name.lower() or "e5-" in self.model_name.lower():
            self.query_prefix = current_app.config.get('KB_EMBEDDING_QUERY_PREFIX', "Represent this sentence for searching relevant passages: ") # BGE-style query
            self.passage_prefix = current_app.config.get('KB_EMBEDDING_PASSAGE_PREFIX', "") # Default to no prefix for BGE passages
            if "e5-" in self.model_name.lower() and not self.passage_prefix: # E5 specific default
                self.passage_prefix = "passage: "
        else: # For other models like all-MiniLM-L6-v2
            self.query_prefix = ""
            self.passage_prefix = ""
        logger.info(f"EmbeddingService using query_prefix: '{self.query_prefix}', passage_prefix: '{self.passage_prefix}'")


    def _initialize_client(self):
        try:
            logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
            self.client = STModel(self.model_name)
            self._dimension = self.client.get_sentence_embedding_dimension()
            logger.info(f"Embedding model '{self.model_name}' loaded. Dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            self.client = None
            self._dimension = 0
            # Consider raising an error here to halt app startup if embeddings are critical
            # raise RuntimeError(f"Failed to load critical embedding model: {self.model_name}") from e

    def get_embeddings(self, texts: Union[str, List[str]], input_type: str = "passage") -> List[List[float]]:
        if not self.client:
            logger.error("EmbeddingService client not initialized. Cannot generate embeddings.")
            num_texts = 1 if isinstance(texts, str) else len(texts)
            return [[] for _ in range(num_texts)]

        if not texts:
            return []

        # Cache check - Suggested Enhancement
        if self.enable_cache and self.cache is not None and isinstance(texts, str):
            # Only cache single string inputs for simplicity, batch caching can be complex
            cache_key = f"{self.model_name}:{input_type}:{texts[:256]}" # Truncate long texts for key
            if cache_key in self.cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                return self.cache[cache_key]

        input_texts_processed = [texts] if isinstance(texts, str) else texts

        if input_type == "query":
            input_texts_processed = [self.query_prefix + text for text in input_texts_processed]
        elif input_type == "passage":
            input_texts_processed = [self.passage_prefix + text for text in input_texts_processed]
        
        try:
            embeddings_np = self.client.encode(input_texts_processed, normalize_embeddings=self.normalize_embeddings)
            result_list = embeddings_np.tolist()
            
            # Cache store - Suggested Enhancement
            if self.enable_cache and self.cache is not None and isinstance(texts, str):
                self.cache[cache_key] = result_list # Store the list containing the single embedding
                logger.debug(f"Cached result for key: {cache_key}")

            return result_list
        except Exception as e:
            logger.error(f"Error during batch embedding generation with '{self.model_name}': {e}", exc_info=True)
            return [[] for _ in input_texts_processed]


    def get_embedding_dimension(self) -> int:
        return self._dimension

# Global instance, initialized in app/__init__.py
embedding_service_instance: Optional[EmbeddingService] = None

def initialize_embedding_service(app): # Pass Flask app
    global embedding_service_instance
    if embedding_service_instance is None:
        try:
            embedding_service_instance = EmbeddingService(
                model_name=app.config.get('KB_EMBEDDING_MODEL_NAME'),
                normalize=app.config.get('KB_EMBEDDING_NORMALIZE', True)
            )
            logger.info("EmbeddingService initialized globally.")
        except Exception as e:
            logger.error(f"Failed to create global EmbeddingService instance: {e}", exc_info=True)
            # App might still start, but KB functionality will be severely impacted.
