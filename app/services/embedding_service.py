import os
import sys
import logging
import traceback
from typing import List, Union, Optional
import numpy as np
from flask import current_app
from sentence_transformers import SentenceTransformer as STModel

# Direct prints for critical startup debugging
print("--- EMBEDDING_SERVICE_DEBUG: Module loading ---")
print(f"--- Python executable: {sys.executable}")
print(f"--- Current working directory: {os.getcwd()}")
print(f"--- Python path: {sys.path}")

try:
    # Try to get the sentence-transformers version
    from sentence_transformers import __version__ as st_version
    print(f"--- sentence-transformers version: {st_version}")
except Exception as e:
    print(f"--- ERROR getting sentence-transformers version: {e}")

logger = logging.getLogger(__name__)
print(f"--- Logger name: {logger.name} (level: {logging.getLevelName(logger.level)})")
print("--- EMBEDDING_SERVICE_DEBUG: Module imports complete ---")

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
        print(f"\n--- EMBEDDING_SERVICE_DEBUG: _initialize_client() called for model: {self.model_name} ---")
        print(f"--- Current working directory: {os.getcwd()}")
        print(f"--- Model name: {self.model_name}")
        
        try:
            print(f"--- Attempting to import SentenceTransformer...")
            from sentence_transformers import __version__ as st_version
            print(f"--- sentence-transformers version: {st_version}")
            
            print(f"--- Creating SentenceTransformer instance for '{self.model_name}'...")
            self.client = STModel(self.model_name)
            print("--- SentenceTransformer instance created successfully!")
            
            print("--- Getting embedding dimension...")
            self._dimension = self.client.get_sentence_embedding_dimension()
            print(f"--- Success! Model loaded with dimension: {self._dimension}")
            
            logger.info(f"Embedding model '{self.model_name}' loaded. Dimension: {self._dimension}")
            return True
            
        except ImportError as ie:
            print(f"--- CRITICAL: Failed to import sentence_transformers: {ie}")
            print("--- Please install it with: pip install -U sentence-transformers")
            print(traceback.format_exc())
            
        except Exception as e:
            print(f"--- CRITICAL: Failed to initialize model '{self.model_name}': {str(e)}")
            print("--- Full traceback:")
            print(traceback.format_exc())
            
        # If we get here, initialization failed
        self.client = None
        self._dimension = 0
        return False

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

def initialize_embedding_service(app):
    global embedding_service_instance
    
    print("\n" + "="*80)
    print("EMBEDDING_SERVICE_DEBUG: initialize_embedding_service() called")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Module file: {__file__}")
    print(f"Initial global embedding_service_instance: {id(embedding_service_instance) if embedding_service_instance is not None else 'None'}")
    
    if embedding_service_instance is not None:
        print("--- Global instance already exists, skipping initialization")
        print(f"--- Instance ID: {id(embedding_service_instance)}")
        print(f"--- Client loaded: {embedding_service_instance.client is not None}")
        return
    
    temp_instance = None
    try:
        model_name = app.config.get('KB_EMBEDDING_MODEL_NAME')
        normalize = app.config.get('KB_EMBEDDING_NORMALIZE', True)
        
        print(f"\n--- Creating temporary EmbeddingService instance")
        print(f"--- Model: {model_name}")
        print(f"--- Normalize: {normalize}")
        
        # Create a temporary instance first
        temp_instance = EmbeddingService(model_name=model_name, normalize=normalize)
        
        # Verify the client was initialized
        if temp_instance.client is None:
            print("\n!!! CRITICAL: Client failed to initialize in temporary instance")
            print("!!! This suggests the model failed to load in _initialize_client()")
            print(f"!!! Model: {model_name}")
            return
            
        # If we got here, everything is good with the temp instance
        print(f"\n--- Temporary instance created successfully")
        print(f"--- Temp instance ID: {id(temp_instance)}")
        print(f"--- Client loaded: {temp_instance.client is not None}")
        
        # Now assign to global
        embedding_service_instance = temp_instance
        print("\n--- Assigned temporary instance to global embedding_service_instance")
        print(f"--- Global instance ID: {id(embedding_service_instance)}")
        print(f"--- Global client valid: {embedding_service_instance.client is not None}")
        
    except Exception as e:
        print(f"\n!!! CRITICAL: Exception during initialization: {str(e)}")
        print("!!! Full traceback:")
        traceback.print_exc()
        if temp_instance:
            embedding_service_instance = None
        return
    
    # Final verification
    if embedding_service_instance and embedding_service_instance.client:
        print("\n--- EMBEDDING SERVICE INITIALIZATION SUCCESSFUL ---")
        print(f"--- Final global instance ID: {id(embedding_service_instance)}")
        print(f"--- Client dimension: {getattr(embedding_service_instance, '_dimension', 'N/A')}")
    else:
        print("\n!!! EMBEDDING SERVICE INITIALIZATION FAILED !!!")
        print("!!! The global instance is either None or has no client !!!")
