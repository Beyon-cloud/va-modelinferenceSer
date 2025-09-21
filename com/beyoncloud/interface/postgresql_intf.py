from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from asyncpg import Record
from com.beyoncloud.db.table_data_model import DynSelectEntity


class PostgreSqlInterface(ABC):

    @abstractmethod
    async def dyn_insert(self, table_name: str, data: List[Dict[str, Any]]) -> Optional[Record]:
        pass

    @abstractmethod
    async def dyn_param_select(self, table_name: str,column_names: List[str], conditions: Dict[str, Any], top_k: int = 10, order_by: Dict[str, List[str]] = None
    ) -> List[Record]:
        pass

    @abstractmethod
    async def dyn_entity_select(self, dyn_select_entity: DynSelectEntity) -> List[Record]:
        pass

    @abstractmethod
    async def dyn_delete(self, table_name: str, conditions: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    async def dyn_update(self, table_name: str,conditions: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    async def dyn_vector_search(self, table_name: str = "", vector_column_name: str = "", conditions: Dict[str, Any] = None, top_k: int = 10, order_by: Dict[str, List[str]] = None) -> Optional[Record]:
        pass