from com.beyoncloud.db.sql_db_connectivity import SqlDbConnectivity

# Create a shared instance
sql_db_connection = SqlDbConnectivity()

def get_sql_db_connection():
    return sql_db_connection
