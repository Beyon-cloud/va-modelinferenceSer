import pandas as pd
import json
import os
from typing import List, Dict, Any
from weaviate_client import WeaviateClient
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, weaviate_client: WeaviateClient):
        self.weaviate_client = weaviate_client
    
    def load_sample_data(self):
        """Load sample data into the knowledge base."""
        sample_documents = [
            {
                "title": "Introduction to Vector Databases",
                "content": """Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. 
                They are particularly useful for machine learning applications, especially in the context of similarity search, 
                recommendation systems, and natural language processing. Vector databases use various indexing techniques like 
                HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) to enable fast similarity searches across 
                millions or billions of vectors.""",
                "category": "technology",
                "source": "documentation"
            },
            {
                "title": "Weaviate Features",
                "content": """Weaviate is an open-source vector database that combines object storage with vector search capabilities. 
                Key features include: automatic vectorization using various ML models (OpenAI, Cohere, Hugging Face), 
                hybrid search combining vector and keyword search, built-in multi-tenancy, replication for high availability, 
                GraphQL and REST APIs, and integration with popular ML frameworks. It supports both cloud and on-premises 
                deployments.""",
                "category": "weaviate",
                "source": "product_info"
            },
            {
                "title": "RAG Architecture",
                "content": """Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with 
                text generation. The process involves: 1) Retrieving relevant documents from a knowledge base using vector 
                similarity search, 2) Providing these documents as context to a language model, 3) Generating responses 
                based on both the retrieved context and the model's training. This approach helps reduce hallucinations 
                and provides more factual, contextual responses.""",
                "category": "ai_concepts",
                "source": "research"
            },
            {
                "title": "Python Client Usage",
                "content": """The Weaviate Python client provides easy integration with Weaviate databases. Basic usage includes: 
                connecting to a Weaviate instance, creating collections with schemas, inserting objects with automatic 
                vectorization, performing vector similarity searches, and executing hybrid queries. The v4 client offers 
                improved performance and a more intuitive API compared to earlier versions.""",
                "category": "programming",
                "source": "tutorial"
            },
            {
                "title": "Best Practices for Vector Search",
                "content": """For optimal vector search performance: 1) Choose appropriate embedding models for your data type, 
                2) Normalize vectors when using cosine similarity, 3) Use appropriate distance metrics (cosine, dot product, euclidean), 
                4) Consider chunking strategies for long documents, 5) Implement proper data preprocessing and cleaning, 
                6) Monitor and tune search parameters based on your specific use case, 7) Use hybrid search for better 
                recall when combining semantic and keyword matching.""",
                "category": "best_practices",
                "source": "guidelines"
            }
        ]
        
        try:
            for doc in sample_documents:
                self.weaviate_client.add_document(
                    content=doc["content"],
                    title=doc["title"],
                    category=doc["category"],
                    source=doc["source"]
                )
            
            logger.info(f"Loaded {len(sample_documents)} sample documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return False
    
    def load_from_csv(self, csv_file: str, content_column: str, title_column: str, 
                      category_column: str = None) -> bool:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                self.weaviate_client.add_document(
                    content=str(row[content_column]),
                    title=str(row[title_column]),
                    category=str(row[category_column]) if category_column else "imported",
                    source="csv_import"
                )
            
            logger.info(f"Loaded {len(df)} documents from CSV")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return False
    
    def load_from_json(self, json_file: str) -> bool:
        """Load data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    self.weaviate_client.add_document(
                        content=item.get("content", ""),
                        title=item.get("title", "Untitled"),
                        category=item.get("category", "imported"),
                        source="json_import"
                    )
                    
                logger.info(f"Loaded {len(data)} documents from JSON")
            else:
                logger.error("JSON file should contain a list of documents")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            return False
    
    def load_text_files(self, directory: str) -> bool:
        """Load all text files from a directory."""
        try:
            count = 0
            for filename in os.listdir(directory):
                if filename.endswith(('.txt', '.md')):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self.weaviate_client.add_document(
                        content=content,
                        title=filename,
                        category="text_files",
                        source="file_import"
                    )
                    count += 1
            
            logger.info(f"Loaded {count} text files")
            return True
            
        except Exception as e:
            logger.error(f"Error loading text files: {e}")
            return False