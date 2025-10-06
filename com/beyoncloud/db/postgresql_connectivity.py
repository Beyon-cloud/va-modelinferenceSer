import logging
import asyncpg
import asyncio
from sqlalchemy import event
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import MetaData
import com.beyoncloud.config.settings.env_config as config

logger = logging.getLogger(__name__)

class PostgreSqlConnectivity:
    _instance = None  # Class-level singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostgreSqlConnectivity, cls).__new__(cls)
            cls._instance.pool = None
            cls._instance.engine = None
            cls._instance.async_session = None
            cls._instance.postgres_uri = config.POSTGRES_URI
        return cls._instance

    async def init_connection(self):
        logger.info("... PostgreSQL DB connectivity initilization ...")
        if self.pool is None:
            try:
                logger.debug(f"Connecting to PostgreSQL using URI: {self.postgres_uri}")
                if not self.postgres_uri:
                    raise ValueError("POSTGRES_URI is not defined in configuration.")

                self.pool = await asyncpg.create_pool(
                    dsn=self.postgres_uri,
                    min_size=int(config.POSTGRES_MINCONN),
                    max_size=int(config.POSTGRES_MAXCONN),
                    command_timeout=config.POSTGRES_CMD_TIMEOUT
                )
                logger.info("PostgreSQL connection pool created successfully.")

                # Init SQLAlchemy async engine and sessionmaker
                self.base = automap_base()
                self.metadata = MetaData()
                sqlalchemy_uri = self.postgres_uri.replace("postgresql://", "postgresql+asyncpg://")
                self.engine = create_async_engine(
                    sqlalchemy_uri,
                    echo=config.SQLALCHEMY_ECHO,  # Optional: enable SQL logging
                    future=True,
                    pool_pre_ping=True,
                    connect_args={"command_timeout": config.POSTGRES_CMD_TIMEOUT}
                )

                self.async_session = async_sessionmaker(
                    bind=self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

                # Initilize automap Base
                await self.prepare_automap()

                logger.info("SQLAlchemy async engine and session created.")
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL connection: {e}")
                raise

    # Prepare base on startup
    async def prepare_automap(self):
        async with self.engine.begin() as conn:
            # Reflect metadata for later use (optional)
            await conn.run_sync(self.metadata.reflect)

            # Automap inside a run_sync block
            def do_prepare(sync_conn):
                self.base.prepare(sync_conn, reflect=True)

            await conn.run_sync(do_prepare)

    async def get_connection(self):
        if not self.pool:
            raise RuntimeError("Connection pool is not initialized. Call init_connection() first.")
        return await self.pool.acquire()

    async def release_connection(self, conn):
        if self.pool and conn:
            await self.pool.release(conn)

    @asynccontextmanager
    async def connection(self):
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.release_connection(conn)

    @asynccontextmanager
    async def orm_session(self):
        """Context manager for SQLAlchemy ORM session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close_connection(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL asyncpg connection pool closed during application shutdown.")

        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.async_session = None
            logger.info("SQLAlchemy engine disposed during application shutdown.")
