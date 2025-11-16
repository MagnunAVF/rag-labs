"""Document ingestion script for RAG system using Weaviate and LlamaIndex."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import weaviate
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.client import WeaviateClient

WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
TEI_BASE_URL = "http://localhost:8082"
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
COLLECTION_NAME = "LlamaIndex"
DATA_DIR = Path("./data/texts")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
MAX_RETRIES = 5
RETRY_DELAY = 2
TEI_TIMEOUT = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def connect_to_weaviate(
    host: str = WEAVIATE_HOST,
    port: int = WEAVIATE_PORT,
    max_retries: int = MAX_RETRIES,
    delay: int = RETRY_DELAY,
) -> WeaviateClient:
    """Connect to Weaviate with retry logic.
    
    Args:
        host: Weaviate host address
        port: Weaviate port number
        max_retries: Maximum number of connection attempts
        delay: Delay in seconds between retries
        
    Returns:
        Connected Weaviate client
        
    Raises:
        SystemExit: If connection fails after all retries
    """
    logger.info("Connecting to Weaviate at %s:%d...", host, port)
    
    for attempt in range(max_retries):
        try:
            client = weaviate.connect_to_local(port=port)
            if client.is_ready():
                logger.info(
                    "Weaviate connection successful (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )
                return client
            
            logger.warning(
                "Weaviate not ready yet (attempt %d/%d)",
                attempt + 1,
                max_retries,
            )
            client.close()
            
        except Exception as e:
            logger.error(
                "Connection attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                str(e),
            )
        
        if attempt < max_retries - 1:
            logger.info("Retrying in %d seconds...", delay)
            time.sleep(delay)
    
    logger.error("Failed to connect to Weaviate after %d attempts", max_retries)
    logger.error("Try restarting Weaviate: docker-compose restart weaviate")
    sys.exit(1)


def configure_settings(
    embed_model_name: str = EMBED_MODEL_NAME,
    tei_base_url: str = TEI_BASE_URL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """Configure LlamaIndex settings for embeddings and text splitting.
    
    Args:
        embed_model_name: Name of the embedding model
        tei_base_url: Base URL for Text Embeddings Inference service
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Number of overlapping tokens between chunks
    """
    logger.info("Configuring LlamaIndex settings...")
    
    Settings.embed_model = TextEmbeddingsInference(
        model_name=embed_model_name,
        base_url=tei_base_url,
        timeout=TEI_TIMEOUT,
        truncate_text=True,
    )
    
    Settings.text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    Settings.llm = None
    logger.info("Settings configured successfully")


def load_documents(data_dir: Path = DATA_DIR) -> list:
    """Load documents from the specified directory.
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of loaded documents
        
    Raises:
        SystemExit: If no documents found or loading fails
    """
    logger.info("Loading documents from %s...", data_dir)
    
    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)
    
    try:
        documents = SimpleDirectoryReader(str(data_dir)).load_data()
        
        if not documents:
            logger.warning("No documents found in %s", data_dir)
            logger.info("Please add files to ingest and try again")
            sys.exit(0)
        
        logger.info("Loaded %d document(s)", len(documents))
        return documents
        
    except Exception as e:
        logger.error("Failed to read documents: %s", str(e))
        sys.exit(1)


def ingest_documents(
    client: WeaviateClient,
    documents: list,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """Ingest documents into Weaviate vector store.
    
    Args:
        client: Connected Weaviate client
        documents: List of documents to ingest
        collection_name: Name of the Weaviate collection
        
    Returns:
        Created vector store index
        
    Raises:
        Exception: If indexing fails
    """
    logger.info("Indexing documents into collection '%s'...", collection_name)
    logger.info("This may take a moment...")
    
    try:
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        
        logger.info(
            "Successfully ingested data into Weaviate collection: '%s'",
            collection_name,
        )
        return index
        
    except Exception as e:
        logger.error("Error during indexing: %s", str(e))
        raise


def main() -> None:
    """Main execution function for document ingestion."""
    client: Optional[WeaviateClient] = None
    
    try:
        client = connect_to_weaviate()
        configure_settings()
        documents = load_documents()
        ingest_documents(client, documents)

        logger.info("Document ingestion completed successfully")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", str(e))
        sys.exit(1)
    
    finally:
        if client is not None:
            client.close()
            logger.info("Weaviate connection closed")


if __name__ == "__main__":
    main()