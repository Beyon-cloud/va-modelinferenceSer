# com/beyoncloud/repositories/postgres_repository.py

import logging
import operator
from traceback import print_exception
from typing import Dict, List, Optional, Union, Any

from sqlalchemy import select, update, and_, or_, inspect, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import ColumnProperty, InstrumentedAttribute

from com.beyoncloud.db.db_connection import sql_db_connection
from com.beyoncloud.common.constants import RepositoryKeys, RepositoryOps
from com.beyoncloud.interface.repository_interface import BaseRepository
from com.beyoncloud.config.settings.table_mapper_config import load_table_mapping  # ← Add this

logger = logging.getLogger(__name__)

# Load mapper globally
mapper = load_table_mapping()


class PostgreSqlRepository(BaseRepository):
    """
    Generic PostgreSQL Repository supporting CRUD, Upsert, and filtered updates.
    Works with any automapped model using logical-to-physical column mapping.
    """

    def __init__(self, model,table_key):
        self.model = model
        self.pg_conn = sql_db_connection.get_postgresql_client()

        # Infer schema and table from model (you can customize this)
        self.schema = "schema1"
        self.table_key = table_key

    # ---------- CRUD ----------
    async def find_by_id(self, id_: int):
        """Fetch a record by primary key."""
        async with self.pg_conn.orm_session() as session:
            return await session.get(self.model, id_)

    async def delete_by_id(self, id_: int):
        """Delete a record by primary key."""
        async with self.pg_conn.orm_session() as session:
            obj = await session.get(self.model, id_)
            if obj:
                await session.delete(obj)
                await session.commit()

    async def find_by_filters(
        self,
        filters: List[Dict],
        order_by: Optional[Dict] = None,
        limit: Optional[int] = 1
    ):
        """
        Fetch records based on filters.
        Args:
            filters: List of filter dicts with logical field names.
            order_by: Optional dict with logical field & direction.
            limit: Number of records to fetch.
        """

        print("find_by_filters")
        async with self.pg_conn.orm_session() as session:
            conditions = self._parse_conditions(filters)
            stmt = select(self.model).where(and_(*conditions))

            if order_by:
                field_name = order_by[RepositoryKeys.FIELD]
                direction = order_by[RepositoryKeys.DIRECTION].lower()
                col_attr = self._get_column_attr(field_name)
                stmt = stmt.order_by(col_attr.desc() if direction == RepositoryKeys.DESC else col_attr.asc())

            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return result.scalars().all() if limit != 1 else result.scalar_one_or_none()

    # ---------- UPSERT ----------
    async def upsert(
        self,
        data_obj: Dict[str, Any],  # keys are logical field names
        conflict_fields: List[str],  # logical names
        update_fields: List[str],   # logical names
        return_field: Optional[str] = None  # logical name
    ) -> Optional[int]:
        """
        Insert or update record on conflict.
        Args:
            data_obj: Dict with logical field names → values
            conflict_fields: List of logical field names for ON CONFLICT
            update_fields: List of logical field names to SET on conflict
            return_field: Logical field to return (e.g., 'id')
        """
        try:
            async with self.pg_conn.orm_session() as session:
                # Resolve logical → physical
                physical_data = {
                    self._get_db_column(logical): value
                    for logical, value in data_obj.items()
                    if self._get_db_column(logical)  # Skip invalid
                }

                conflict_cols = [self._get_column_attr(f) for f in conflict_fields]
                update_data = {
                    self._get_column_attr(f): physical_data[self._get_db_column(f)]
                    for f in update_fields
                    if self._get_db_column(f) in physical_data
                }

                # Add timestamps
                now = func.now()
                if "updated_at" in {c.key for c in self.model.__table__.columns}:
                    update_data[self.model.updated_at] = now

                stmt = (
                    pg_insert(self.model)
                    .values(**physical_data)
                    .on_conflict_do_update(
                        index_elements=conflict_cols,
                        set_=update_data
                    )
                )

                if return_field:
                    return_col = self._get_column_attr(return_field)
                    stmt = stmt.returning(return_col)
                else:
                    pk_col = inspect(self.model).primary_key[0]
                    stmt = stmt.returning(pk_col)

                result = await session.execute(stmt)
                await session.commit()

                scalars = result.scalars().all()
                return scalars[0] if scalars else None

        except Exception as e:
            logger.exception("Upsert failed for model=%s, error=%s", self.model.__name__, str(e))
            raise

    # ---------- UPDATE ----------
    async def update_by_filters(
        self,
        filters: List[Dict],
        update_fields: Dict[str, Any],  # logical_name → value
        return_field: Optional[str] = None,
        return_all: bool = False,
    ) -> Union[Optional[Any], List[Any]]:
        """
        Update records matching filters.
        Args:
            filters: List of filters with logical field names.
            update_fields: Dict of logical field → new value.
            return_field: Logical field to return (e.g., 'id').
            return_all: Return list or first value.
        """
        try:
            async with self.pg_conn.orm_session() as session:
                conditions = self._parse_conditions(filters)

                # Convert update_fields: logical → physical ORM attr
                physical_updates = {
                    self._get_column_attr(k): v for k, v in update_fields.items()
                }

                # Add timestamps
                now = func.now()
                if hasattr(self.model, "updated_at"):
                    physical_updates[self.model.updated_at] = now
                
                # Resolve return column
                if return_field:
                    return_col = self._get_column_attr(return_field)
                else:
                    return_col = inspect(self.model).primary_key[0]

                stmt = (
                    update(self.model.__table__)
                    .where(and_(*conditions))
                    .values(physical_updates)
                    .returning(return_col)
                )

                result = await session.execute(stmt)
                await session.commit()

                values = result.scalars().all()
                if return_all:
                    return values
                return values[0] if values else None

        except Exception as e:
            logger.exception("Update failed for model=%s", self.model.__name__)
            raise

    # ---------- HELPERS ----------
    def _get_db_column(self, logical_name: str) -> Optional[str]:
        """Resolve logical field name → DB column name."""
        return mapper.get_db_column_name(self.schema, self.table_key, logical_name)

    def _get_column_attr(self, logical_name: str) -> InstrumentedAttribute:
        """Resolve logical field name → ORM column attribute."""
        db_col = self._get_db_column(logical_name)
        if not db_col:
            raise ValueError(f"Unknown logical field: '{logical_name}' in {self.table_key}")
        if not hasattr(self.model, db_col):
            raise AttributeError(f"Model '{self.model.__name__}' has no column '{db_col}'")
        return getattr(self.model, db_col)

    def _parse_conditions(self, filters: List[Dict]) -> List:
        """Parse filters with logical field names."""
        def parse_item(item: Dict):
            logical_field = item[RepositoryKeys.FIELD]
            col_attr = self._get_column_attr(logical_field)
            op = item[RepositoryKeys.OPERATION]
            value = item[RepositoryKeys.VALUE]
            return self._get_operator_map()[op](col_attr, value)

        conditions = []
        for f in filters:
            if RepositoryKeys.LOGIC in f:
                nested = self._parse_conditions(f[RepositoryKeys.CONDITIONS])
                if f[RepositoryKeys.LOGIC] == RepositoryKeys.AND:
                    conditions.append(and_(*nested))
                else:
                    conditions.append(or_(*nested))
            else:
                conditions.append(parse_item(f))
        return conditions

    @staticmethod
    def _get_operator_map():
        return {
            RepositoryOps.EQUALS: operator.eq,
            RepositoryOps.NOT_EQUALS: operator.ne,
            RepositoryOps.LESS_THAN: operator.lt,
            RepositoryOps.LESS_THAN_EQUALS: operator.le,
            RepositoryOps.GREATER_THAN: operator.gt,
            RepositoryOps.GREATER_THAN_EQUALS: operator.ge,
            RepositoryOps.IN: lambda f, v: f.in_(v),
            RepositoryOps.LIKE: lambda f, v: f.like(v),
            RepositoryOps.ILIKE: lambda f, v: f.ilike(v),
            RepositoryOps.IS: lambda f, v: f.is_(v),
        }