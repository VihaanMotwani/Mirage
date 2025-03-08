import os
import logging
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create connection pool
connection_pool = None

try:
    # Parse the DATABASE_URL to extract connection parameters
    # Format: postgresql://username:password@hostname:port/database_name
    if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
        # Remove the protocol part
        db_url = DATABASE_URL.replace("postgresql://", "")
        
        # Split the remaining string by @ to separate credentials and host
        credentials_host = db_url.split("@")
        
        if len(credentials_host) == 2:
            # Extract username and password
            user_pass = credentials_host[0].split(":")
            username = user_pass[0]
            password = user_pass[1] if len(user_pass) > 1 else ""
            
            # Extract host, port, and database name
            host_port_db = credentials_host[1].split("/")
            host_port = host_port_db[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            database = host_port_db[1] if len(host_port_db) > 1 else ""
            
            # Create a connection pool
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                user=username,
                password=password,
                host=host,
                port=port,
                database=database
            )
            
            logger.info(f"Database connection pool created successfully to {host}:{port}/{database}")
        else:
            logger.error("Invalid DATABASE_URL format")
    else:
        logger.error("DATABASE_URL environment variable not found or invalid")
except Exception as e:
    logger.error(f"Error creating database connection pool: {str(e)}")

def get_db_connection():
    """
    Get a connection from the pool.
    
    Returns:
        Connection object or None if pool not initialized
    """
    if connection_pool:
        try:
            return connection_pool.getconn()
        except Exception as e:
            logger.error(f"Error getting database connection: {str(e)}")
            return None
    else:
        logger.error("Connection pool not initialized")
        return None

def release_db_connection(conn):
    """
    Release a connection back to the pool.
    
    Args:
        conn: Connection object to release
    """
    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing database connection: {str(e)}")