import logging
from typing import List
from pathlib import Path
import com.beyoncloud.config.settings.env_config as config
from com.beyoncloud.weaviate.weaviate_client import WeaviateClient
from com.beyoncloud.datamodel.weaviate_data_model import KnowledgeBase,SearchBase, FileBase

logger = logging.getLogger(__name__)

def save_text(knowledge_base: List[KnowledgeBase]):

    # Initialize Weaviate client (local)
    print("🔌 Connecting to local Weaviate...")
    weaviate_client = WeaviateClient()
    weaviate_client.create_knowledge_base_schema()
    print("✅ Connected successfully!")

    if knowledge_base:
        for record in knowledge_base:
            result = weaviate_client.add_text(
                content=record.content,
                title=record.title,
                category=record.category,
                source=record.source
            )

    return result

def search_text(search_base: SearchBase):
    weaviate_client = WeaviateClient()
    limit = 5
    if "hybrid" == search_base.search_type:
        result = weaviate_client.hybrid_search(search_base.query, limit)
    else:
        result = weaviate_client.search_similar(search_base.query, limit)
    return result


async def save_file(file_base: FileBase):
    path = Path(file_base.file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_base.file_path} not found")

    weaviate_client = WeaviateClient()
    result = weaviate_client.save_file1(file_base.file_path, )
    weaviate_client.close()
    return result

async def search_by_file(search_base: SearchBase):
    weaviate_client = WeaviateClient()
    response = weaviate_client.search_by_file1(search_base.query, config.MODEL_CONFIG.get("query_retrieval_top_k"))
    return response