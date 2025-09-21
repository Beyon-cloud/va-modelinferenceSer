import logging
from typing import Dict,Any, List, Optional, Union, Tuple
from asyncpg import Record
from com.beyoncloud.interface.postgresql_intf import PostgreSqlInterface
from com.beyoncloud.db.postgresql_connectivity import PostgreSqlConnectivity
from com.beyoncloud.db.table_data_model import DynSelectEntity
import com.beyoncloud.config.settings.env_config as config
from sqlalchemy import select, Column, Integer, String, JSON, and_, asc, desc, join, func

logger = logging.getLogger(__name__)


class PostgresSqlImpl(PostgreSqlInterface):
    """
    Implementation of PostgreSqlInterface for interacting with a PostgreSQL database asynchronously.

    This class provides dynamic methods for common database operations such as insert, select,
    update, delete, and vector-based similarity search. It constructs SQL queries dynamically based on
    provided parameters and entities, and executes them using an asynchronous PostgreSQL client.

    Attributes:
        postgres_client (PostgreSqlConnectivity): An instance managing PostgreSQL connections.

    Methods:
        dyn_insert(table_name: str, data: List[Dict[str, Any]]) -> Optional[Record]:
            Inserts multiple records into the specified table dynamically.

        dyn_param_select(table_name: str, column_names: List[str], conditions: Dict[str, Any], top_k: int = 10, order_by: Dict[str, List[str]] = None) -> List[Record]:
            Selects records from a table based on specified columns, conditions, ordering, and limit.

        dyn_entity_select(dyn_select_entity: DynSelectEntity) -> List[Record]:
            Selects records using a DynSelectEntity object containing query parameters.

        dyn_delete(table_name: str, conditions: Dict[str, Any]) -> int:
            Deletes records matching the given conditions from the specified table.

        dyn_update(table_name: str, conditions: Dict[str, Any], update_data: Dict[str, Any]) -> int:
            Updates records matching conditions with the provided data.

        dyn_vector_search(table_name: str, vector_column_name: str, conditions: Dict[str, Any], top_k: int = 10, order_by: Dict[str, List[str]] = None) -> Optional[Record]:
            Performs a vector similarity search on a specified vector column with optional conditions and ordering.

        dict_to_conditions_clause(conditions: Dict[str, Any]) -> str:
            Converts a dictionary of conditions into an SQL WHERE clause string.

        dict_to_order_clause(order_dict: Dict[str, List[str]]) -> str:
            Converts a dictionary defining order directions and columns into an SQL ORDER BY clause string.

        dict_to_set_clause(conditions: Dict[str, Any]) -> str:
            Converts a dictionary into an SQL SET clause string for update queries.

    Author: Jenson
    Date: 03-June-2025 
    """

    def __init__(self):
        logger.info("Initializing PostgreSqlImpl...")
        self.postgres_client = PostgreSqlConnectivity()

    # Method using for dynamically insert the record
    async def dyn_insert(self, table_name: str, data: List[Dict[str, Any]]) -> Optional[Record]:
        """
        Inserts multiple records dynamically into the specified table.

        Args:
            table_name (str): The name of the table where records will be inserted.
            data (List[Dict[str, Any]]): A list of dictionaries representing the records to insert.

        Returns:
            Optional[Record]: The result of the insert operation, or None.
        """

        for record in data:
            record_dict = record.dict()
            keys = ', '.join(record_dict.keys())
            values_placeholders = ', '.join(f'${i + 1}' for i in range(len(record_dict)))
            values = list(record_dict.values())

            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_insert_query']
            query = query_template.format(
                table=table_name,
                keys=keys,
                values_placeholders=values_placeholders
            )
            print("query : "+query)
            logger.debug(f"Create query: {query} with values: {values}")
            async with self.postgres_client.connection() as conn:
                await conn.fetchrow(query, *values)

    # Method using for dynamically fetch the record based on parameter
    async def dyn_param_select(self, table_name: str,column_names: List[str], conditions: Dict[str, Any], top_k: int = 10, order_by: Dict[str, List[str]] = None) -> List[Record]:
        """
        Fetches records dynamically from a table based on given parameters.

        Args:
            table_name (str): Name of the table to query.
            column_names (List[str]): Columns to select; use ['*'] or empty for all.
            conditions (Dict[str, Any]): Conditions for the WHERE clause.
            top_k (int, optional): Maximum number of records to return. Defaults to 10.
            order_by (Dict[str, List[str]], optional): Order by clause as a dictionary. Example: {"ASC": ["id"]}.

        Returns:
            List[Record]: List of matching records.
        """

        if table_name:
            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_select_query']

            if column_names:
                column_name = ', '.join(column_names)
            else: 
                column_name = " * "

            if conditions:
                condition = self.dict_to_conditions_clause(conditions)
            else:
                condition = " 1 = 1 "

            if order_by:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['order_by']
                order_by_str = self.dict_to_order_clause(order_by)
                print("IF")
            else:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['empty_order_by']
                order_by_str = ""
                print("ELSE")
            
            print(orderby_template)
            print(order_by_str)
                
            final_query_template = query_template.format(
                table_name=table_name,
                column_name=column_name,
                condition=condition,
                orderby_template=orderby_template,
                limit=top_k
            )
            if "{order_by}" in final_query_template.lower():
                query = final_query_template.format(
                    order_by = order_by_str
                )
            else:
                query = final_query_template
            print("query : "+query)
            logger.debug(f"Created query: {query}")
            async with self.postgres_client.connection() as conn:
                return await conn.fetch(query)

    # Method using for dynamically fetch the record based on Entity
    async def dyn_entity_select(self, dyn_select_entity: DynSelectEntity) -> List[Record]:
        """
        Fetches records dynamically based on a DynSelectEntity object.

        Args:
            dyn_select_entity (DynSelectEntity): Object containing table name, column names,
                                                 conditions, order by, and top_k limit.

        Returns:
            List[Record]: List of matching records.
        """

        if dyn_select_entity.table_name:
            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_select_query']

            if dyn_select_entity.column_names:
                column_name = ', '.join(dyn_select_entity.column_names)
            else: 
                column_name = " * "

            if dyn_select_entity.conditions:
                condition = self.dict_to_conditions_clause(dyn_select_entity.conditions)
            else:
                condition = " 1 = 1 "

            if dyn_select_entity.order_by:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['order_by']
                order_by_str = self.dict_to_order_clause(dyn_select_entity.order_by)
                print("IF")
            else:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['empty_order_by']
                order_by_str = ""
                print("ELSE")
            
            print(orderby_template)
            print(order_by_str)
                
            final_query_template = query_template.format(
                table_name=dyn_select_entity.table_name,
                column_name=column_name,
                condition=condition,
                orderby_template=orderby_template,
                limit=dyn_select_entity.top_k
            )
            if "{order_by}" in final_query_template.lower():
                query = final_query_template.format(
                    order_by = order_by_str
                )
            else:
                query = final_query_template
            print("query : "+query)
            logger.debug(f"Created query: {query}")
            async with self.postgres_client.connection() as conn:
                return await conn.fetch(query)

    # Method using for dynamically delete the record
    async def dyn_delete(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """
        Deletes records from a specified table based on conditions.

        Args:
            table_name (str): Name of the table from which to delete records.
            conditions (Dict[str, Any]): Conditions to match records for deletion.

        Returns:
            int: Number of records deleted.
        """

        if table_name:
            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_delete_query']

            if conditions:
                condition = self.dict_to_conditions_clause(conditions)
            else:
                condition = " 1 = 1 "

            query = query_template.format(
                table_name=table_name,
                condition=condition
            )
            print("query : "+query)
            logger.debug(f"Created query: {query}")
            async with self.postgres_client.connection() as conn:
                result = await conn.execute(query)
                return int(result.split()[-1])

    # Method using for dynamically update the record
    async def dyn_update(self, table_name: str,conditions: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """
        Updates records in a table that match given conditions.

        Args:
            table_name (str): Name of the table to update.
            conditions (Dict[str, Any]): Conditions to match records for update.
            update_data (Dict[str, Any]): Dictionary of column-value pairs to update.

        Returns:
            int: Number of records updated.
        """

        if table_name and update_data:
            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_update_query']

            if conditions:
                condition = self.dict_to_conditions_clause(conditions)
            else:
                condition = " 1 = 1 "

            if update_data:
                set_clause = self.dict_to_set_clause(update_data)

            query = query_template.format(
                table_name=table_name,
                column_name=set_clause,
                condition=condition
            )
            print("query : "+query)
            logger.debug(f"Created query: {query}")
            async with self.postgres_client.connection() as conn:
                result = await conn.execute(query)
                return int(result.split()[-1])

    # Method using for dynamically fetch the similarity record (Vector based search)
    async def dyn_vector_search(self, table_name: str = "", vector_column_name: str = "", conditions: Dict[str, Any] = None, top_k: int = 10, order_by: Dict[str, List[str]] = None) -> Optional[Record]:
        """
        Performs a vector similarity search on the specified table.

        Args:
            table_name (str): Name of the table to search.
            vector_column_name (str): Name of the vector column to compare.
            conditions (Dict[str, Any], optional): Optional WHERE conditions.
            top_k (int, optional): Number of top results to return. Defaults to 10.
            order_by (Dict[str, List[str]], optional): Optional ordering instructions.

        Returns:
            Optional[Record]: List of similar records, or None.
        """

        if table_name and vector_column_name:
            query_template = config.QUERY_CONFIG['postgres_queries']['dyn_vector_search_query']

            if conditions:
                condition = self.dict_to_conditions_clause(conditions)
            else:
                condition = " 1 = 1 "

            if order_by:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['order_by']
                order_by_str = self.dict_to_order_clause(order_by)
                print("IF")
            else:
                orderby_template=config.QUERY_CONFIG['postgres_queries']['empty_order_by']
                order_by_str = ""
                print("ELSE")
            
            print(orderby_template)
            print(order_by_str)
                
            final_query_template = query_template.format(
                table_name=table_name,
                vector_column_name=vector_column_name,
                condition=condition,
                orderby_template=orderby_template,
                limit=top_k
            )
            if "{order_by}" in final_query_template.lower():
                query = final_query_template.format(
                    order_by = order_by_str
                )
            else:
                query = final_query_template
            print("query : "+query)
            logger.debug(f"Created query: {query}")
            async with self.postgres_client.connection() as conn:
                return await conn.fetch(query)


    # Method using to convert dictionary data to sql condition clause format
    def dict_to_conditions_clause(self, conditions: Dict[str, Any]) -> str:
        """
        Converts a dictionary of conditions into an SQL WHERE clause.

        Args:
            conditions (Dict[str, Any]): Dictionary of conditions.

        Returns:
            str: SQL-compatible WHERE clause string.
        """

        return " and ".join(
            f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}"
            for key, value in conditions.items()
        )

    # Method using to convert dictionary data to sql order by clause format
    def dict_to_order_clause(self, order_dict: Dict[str, List[str]]) -> str:
        """
        Converts a dictionary into an SQL ORDER BY clause.

        Args:
            order_dict (Dict[str, List[str]]): Dictionary with sort direction and columns.

        Returns:
            str: SQL ORDER BY clause string.
        """

        for direction, columns in order_dict.items():
            column_str = ",".join(columns)
        return f"{column_str} {direction}"

    # Method using to convert dictionary data to sql update set clause format
    def dict_to_set_clause(self, conditions: Dict[str, Any]) -> str:
        """
        Converts a dictionary of key-value pairs into an SQL SET clause for update queries.

        Args:
            conditions (Dict[str, Any]): Dictionary of columns and new values.

        Returns:
            str: SQL SET clause string.
        """

        return " , ".join(
            f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}"
            for key, value in conditions.items()
        )
    
    async def sqlalchemy_insert_one(self, record: Any, return_field: str) -> Any:
        if not record:
            return  # nothing to insert

        async with self.postgres_client.orm_session() as session:
            session.add(record)
            await session.flush()  # to get auto-incremented ID like `id`
            
            # Dynamically get the attribute from the record
            result = None
            if return_field:
                result = getattr(record, return_field, None)
            return result

    async def sqlalchemy_insert_many(self, record_list: list[Any]):
        """
        Inserts a batch of SQLAlchemy ORM records into the database.

        This method accepts a list of ORM model instances and performs a bulk insert
        operation within an asynchronous SQLAlchemy session. It automatically commits 
        the transaction upon successful insertion.

        Args:
            record_list (list[Any]): A list of SQLAlchemy ORM model instances 
                                     (e.g., VARagDoc, VARagEmbed). The list must
                                     contain at least one item; otherwise, no operation is performed.

        Raises:
            Exception: Propagates any exceptions raised during the database operation.

        Example:
            records = [
                VARagDoc(vrd_website_name="site", vrd_content="text", vrd_url="https://a.com"),
                VARagDoc(vrd_website_name="site2", vrd_content="text2", vrd_url="https://b.com")
            ]
            await insert_batch_records(records)
        """

        if not record_list:
            return  # nothing to insert

        async with self.postgres_client.orm_session() as session:
            session.add_all(record_list)

    # Method using for fetch the record based on query and param value
    async def query_select(self, queryKey: str, paramValue: List[Any] = []) -> List[Record]:

        query_template = config.QUERY_CONFIG['postgres_queries'][queryKey]
        
        #print("query_template : "+query_template)
        logger.debug(f"Select query: {query_template}")
        async with self.postgres_client.connection() as conn:
            return await conn.fetch(query_template, *paramValue)

    async def sqlalchemy_get_max_seqno(
        self,
        model: Any,  # automap class
        join_models: Optional[List[tuple]] = None,
        filters: Optional[List[Any]] = None,  # list of filter conditions
        column_name: Optional[Any] = None  # Column name to get max seqno
    ):
        """
        Example Usage:
        

        """
        async with self.postgres_client.orm_session() as session:
            # Select entire model or specific columns
            if column_name:
                stmt = select(func.max(column_name))

                # Apply JOINs
                if join_models:
                    for join_model, on_condition in join_models:
                        stmt = stmt.join(join_model, on_condition)
                # Add WHERE clause if filters provided
                if filters:
                    stmt = stmt.where(and_(*filters))

                # Execute and return results
                results = await session.execute(stmt)
                return results.all() if column_name else results.scalars().all()

    async def sqlalchemy_dynamic_select(
        self,
        model: Any,  # automap class
        filters: Optional[List[Any]] = None,  # list of filter conditions
        order_by: Optional[List[Union[Any, tuple]]] = None,  # list of columns or (column, "asc"/"desc")
        column_names: Optional[List[Any]] = None,  # list of columns to select
        limit: Optional[int] = None
    ):
        """
        Example Usage:
        

        """
        async with self.postgres_client.orm_session() as session:
            # Select entire model or specific columns
            if column_names:
                #select_columns = [getattr(model, col) for col in column_names]
                stmt = select(*column_names)
            else:
                stmt = select(model)

            # Add WHERE clause if filters provided
            if filters:
                stmt = stmt.where(and_(*filters))
                #for key, value in filters.items():
                #    stmt = stmt.where(getattr(model, key) == value)

            # Add ORDER BY
            if order_by:
                order_criteria = []
                for item in order_by:
                    if isinstance(item, tuple):
                        col, direction = item
                        order_criteria.append(asc(col) if direction.lower() == "asc" else desc(col))
                    else:
                        order_criteria.append(asc(item))  # default to ASC if not specified
                stmt = stmt.order_by(*order_criteria)

            # Add limit
            if limit:
                stmt = stmt.limit(limit)

            # Execute and return results
            results = await session.execute(stmt)
            return results.all() if column_names else results.scalars().all()


    async def sqlalchemy_dynamic_join_select(
        self,
        base_model: Any,  # main automap model
        join_models: Optional[List[tuple]] = None,  # list of (model_to_join, on_condition)
        filters: Optional[List[Any]] = None,
        order_by: Optional[List[Union[Any, tuple]]] = None,
        select_columns: Optional[List[Any]] = None,
        limit: Optional[int] = None,
        vector_filter: Optional[Tuple[Any, List[float], str, Optional[float]]] = None
    ):
        """
        Example Usage:
        

        """
        async with self.postgres_client.orm_session() as session:
            # Start select statement
            stmt = select(*select_columns) if select_columns else select(base_model)

            # Apply JOINs
            if join_models:
                for join_model, on_condition in join_models:
                    stmt = stmt.join(join_model, on_condition)

            # Add WHERE filters
            if filters:
                stmt = stmt.where(and_(*filters))

            # Apply vector filter
            if vector_filter:
                vec_col, query_vec, method, threshold = vector_filter

                similarity_expr = {
                    "l2": vec_col.l2_distance(query_vec),
                    "cosine": vec_col.cosine_distance(query_vec),
                    "dot": vec_col.max_inner_product(query_vec),
                }.get(method.lower())

                if similarity_expr is None:
                    raise ValueError(f"Unsupported similarity method: {method}")

                if similarity_expr is not None:
                    stmt = stmt.where(similarity_expr <= threshold)

                # Default: order by similarity
                stmt = stmt.order_by(similarity_expr)

            # Apply ORDER BY
            if order_by:
                order_criteria = []
                for item in order_by:
                    if isinstance(item, tuple):
                        col, direction = item
                        order_criteria.append(asc(col) if direction.lower() == "asc" else desc(col))
                    else:
                        order_criteria.append(asc(item))  # default to ASC
                stmt = stmt.order_by(*order_criteria)

            # Apply LIMIT
            if limit:
                stmt = stmt.limit(limit)

            # Execute and return results
            results = await session.execute(stmt)
            return results.all()
