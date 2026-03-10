

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as PGConnection

# Load environment variables from the project .env file
load_dotenv()


# Centralized database configuration
DB_CONFIG = {
    "host": os.environ["POSTGRES_HOST"],
    "port": int(os.environ["POSTGRES_PORT"]),
    "dbname": os.environ["POSTGRES_DB"],
    "user": os.environ["POSTGRES_USER"],
    "password": os.environ["POSTGRES_PASSWORD"],
}


def get_connection() -> PGConnection:
    """
    Create and return a PostgreSQL connection using environment configuration.
    """
    return psycopg2.connect(**DB_CONFIG)


def get_cursor():
    """
    Convenience helper that returns (connection, cursor).
    Useful for simple scripts.
    """
    conn = get_connection()
    return conn, conn.cursor()