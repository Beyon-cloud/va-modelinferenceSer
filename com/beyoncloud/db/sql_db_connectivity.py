from com.beyoncloud.db.postgresql_connectivity import PostgreSqlConnectivity

class SqlDbConnectivity:
    """
        SQL DB Connectivity Module

        This module provides a unified interface to interact with SQL DB's

        Author: Jenson
        Year: 2025
    """
    def __init__(self):
        print("Initializing SQL DB Connectivity...")
        self.postgresql_client = PostgreSqlConnectivity()

    async def initialize(self):
       await self.postgresql_client.init_connection()

    def get_postgresql_client(self):
        return self.postgresql_client

    async def close_connection(self):
        if self.postgresql_client:
            await self.postgresql_client.close_connection()