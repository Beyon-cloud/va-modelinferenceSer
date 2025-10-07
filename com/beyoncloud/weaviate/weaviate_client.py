import weaviate
import base64
from datetime import datetime
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Property, DataType, Configure
import os
from typing import List, Dict, Any
import logging
from pathlib import Path
from PyPDF2 import PdfReader
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaviateClient:
    def __init__(self, url: str = None):
        # Default to local Weaviate
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")

        try:
            # v4 client connection
            #self.client = weaviate.WeaviateClient(
            #    connection_params=ConnectionParams.from_url(self.url)
            #)
            #self.client = weaviate.connect_to_local()
            self.client = weaviate.connect_to_custom(
                http_host="localhost",
                http_port=8080,
                http_secure=False,
                grpc_host="localhost",
                grpc_port=50051,
                grpc_secure=False
            )
            self.create_knowledge_base_schema()
            #self.client.collections.delete(name="FileDoc1")
            self.create_filedoc_schema1()

            logger.info(f"Connected to local Weaviate at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def create_knowledge_base_schema(self):
        """Create the knowledge base schema (using local text2vec-transformers)."""
        try:
            if self.client.collections.exists("KnowledgeBase"):
                logger.info("KnowledgeBase collection already exists")
                return

            self.client.collections.create(
                name="KnowledgeBase",
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                ],
            )
            logger.info("KnowledgeBase collection created successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def create_filedoc_schema(self):
        """Create FileDoc collection (no vectorizer)."""
        try:
            if self.client.collections.exists("FileDoc"):
                logger.info("FileDoc collection already exists")
                return

            self.client.collections.create(
                name="FileDoc",
                vectorizer_config=Configure.Vectorizer.none(),  # no embeddings
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="mimeType", data_type=DataType.TEXT),
                    Property(name="sizeBytes", data_type=DataType.NUMBER),
                    Property(name="textContent", data_type=DataType.TEXT),
                ],
            )
            logger.info("FileDoc collection created successfully")

        except Exception as e:
            logger.error(f"Error creating FileDoc schema: {e}")
            raise

    def add_text(self, content: str, title: str, category: str = "general", source: str = "manual"):
        """Add a document to the knowledge base."""
        try:
            collection = self.client.collections.get("KnowledgeBase")

            result = collection.data.insert({
                "content": content,
                "title": title,
                "category": category,
                "source": source,
                "timestamp": datetime.now(),
            })

            logger.info(f"Document '{title}' added")
            return result

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            collection = self.client.collections.get("KnowledgeBase")

            response = collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["score"],
            )

            return [
                {
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "category": obj.properties.get("category", ""),
                    "source": obj.properties.get("source", ""),
                    "score": obj.metadata.score if obj.metadata else 0.0,
                }
                for obj in response.objects
            ]

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search."""
        try:
            collection = self.client.collections.get("KnowledgeBase")

            response = collection.query.hybrid(
                query=query,
                limit=limit,
                return_metadata=["score"],
            )

            return [
                {
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "category": obj.properties.get("category", ""),
                    "source": obj.properties.get("source", ""),
                    "score": obj.metadata.score if obj.metadata else 0.0,
                }
                for obj in response.objects
            ]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def get_collections_info(self) -> Dict[str, Any]:
        """Get information about available collections."""
        try:
            collections = []
            for collection_name in self.client.collections.list_all():
                collection = self.client.collections.get(collection_name)
                count = collection.aggregate.over_all(total_count=True).total_count
                collections.append({"name": collection_name, "count": count})

            return {"collections": collections}

        except Exception as e:
            logger.error(f"Error getting collections info: {e}")
            return {"collections": []}

    def close(self):
        """Close the connection to Weaviate."""
        if hasattr(self, "client") and self.client:
            self.client.close()
            logger.info("Weaviate connection closed")


    def save_file(self, file_path: str):
        """Save file metadata into FileDoc collection."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        meta = {
            "title": path.stem,
            "filename": path.name,
            "source": "local",
            "path": str(path.resolve()),
            "mimeType": "application/octet-stream",
            "sizeBytes": path.stat().st_size,
            "textContent": "",
        }
        collection = self.client.collections.get("FileDoc")
        obj_id = collection.data.insert(meta)
        print(f"Created FileDoc object with id: {obj_id}")
        return obj_id

    def search_by_file(self, query: str, top_k: int = 5):
        """
        Search FileDoc collection by filename/path/title using BM25 keyword search.
        """
        try:
            collection = self.client.collections.get("FileDoc")

            # BM25 is the replacement for `get + where` in v4
            response = collection.query.bm25(
                query=query,
                limit=top_k,
                return_metadata=["score"],
                return_properties=["title", "filename", "path", "mimeType"]
            )

            results = []
            for obj in response.objects:
                results.append({
                    "id": str(obj.uuid),
                    "title": obj.properties.get("title", ""),
                    "filename": obj.properties.get("filename", ""),
                    "path": obj.properties.get("path", ""),
                    "mimeType": obj.properties.get("mimeType", ""),
                    "score": obj.metadata.score if obj.metadata else 0.0,
                })

            return results

        except Exception as e:
            logger.error(f"Error searching by file: {e}")
            return []


    def search_by_fileX(self, query: str, top_k: int = 5):
        collection = self.client.collections.get("FileDoc")
        response = collection.query.get("FileDoc", ["title", "filename", "path", "mimeType"])\
            .with_where({"path": {"operator": "Like", "valueText": f"%{query}%"}})\
            .with_limit(top_k).do()
        return response

    def create_filedoc_schema1(self):
        """Create FileDoc collection (store binary PDFs)."""
        try:
            if self.client.collections.exists("FileDoc1"):
                logger.info("FileDoc collection already exists")
                return

            self.client.collections.create(
                name="FileDoc1",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="mimeType", data_type=DataType.TEXT),
                    Property(name="sizeBytes", data_type=DataType.NUMBER),
                    Property(name="fileBlob", data_type=DataType.BLOB),        # Raw Base64 file
                    Property(name="fileBlobText", data_type=DataType.TEXT),    # Searchable Base64
                ],
            )

            logger.info("FileDoc collection created successfully")

        except Exception as e:
            logger.error(f"Error creating FileDoc schema: {e}")
            raise


    def save_file1(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        pdf_text = ""
        with open(path, "rb") as f:
            file_bytes = f.read()
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"

        file_b64 = base64.b64encode(file_bytes).decode("utf-8")

        meta = {
            "title": path.stem,
            "filename": path.name,
            "source": "local",
            "path": str(path.resolve()),
            "mimeType": "application/octet-stream",
            "sizeBytes": path.stat().st_size,
            "fileBlob": file_b64,      # for storage
            "fileBlobText": pdf_text,  # for search
        }

        collection = self.client.collections.get("FileDoc1")
        obj_id = collection.data.insert(meta)
        print(f"Created FileDoc1 object with id: {obj_id}")
        return obj_id



    def search_by_file1(self, query: str, top_k: int = 5):
        """Search FileDoc1 collection by filename/title/blobText."""
        try:
            collection = self.client.collections.get("FileDoc1")
            response = collection.query.bm25(
                query=query,
                limit=top_k,
                return_metadata=["score"],
                return_properties=["title", "filename", "path", "mimeType", "fileBlobText"]
            )
            results = []
            for obj in response.objects:
                results.append({
                    "id": str(obj.uuid),
                    "title": obj.properties.get("title", ""),
                    "filename": obj.properties.get("filename", ""),
                    "path": obj.properties.get("path", ""),
                    "mimeType": obj.properties.get("mimeType", ""),
                    "score": obj.metadata.score if obj.metadata else 0.0,
                })
            return results
        except Exception as e:
            logger.error(f"Error searching by file: {e}")
            return []

    def extract_pdf_text(self, file_path: str) -> str:
        """Extract all text from a PDF file."""
        text_content = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return text_content.strip()