from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseRepository(ABC):

    @abstractmethod
    async def find_by_id(self, id_: int):
        pass

    @abstractmethod
    async def find_by_filters(self, filters: List[Dict], order_by: Optional[Dict] = None) -> Optional[object]:
        pass

    @abstractmethod
    async def upsert(self, data_obj, conflict_fields: List[str], update_fields: Optional[List[str]] = None) -> int:
        pass

    @abstractmethod
    async def update_by_filters(self, filters: List[Dict], update_fields: Dict) -> Optional[int]:
        pass

    @abstractmethod
    async def delete_by_id(self, id_: int):
        pass