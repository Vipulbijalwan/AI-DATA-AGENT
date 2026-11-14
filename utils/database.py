import psycopg2



class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = psycopg2.connect(**self.db_config)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()


    def schema_details(self, schema_name):
        if self.cursor is None:
            raise RuntimeError("Database connection has not been opened")

        self.cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (schema_name,)
        )
        tables = self.cursor.fetchall()
        schema_info = {}
        for table in tables:
            table_name = table[0]
            self.cursor.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s",
                (schema_name, table_name)
            )
            schema_info[table_name] = self.cursor.fetchall()
        return schema_info


    def execute_query(self, query):
        if self.cursor is None:
            raise RuntimeError("Database connection has not been opened")

        self.cursor.execute("SET TRANSACTION READ ONLY")
        self.cursor.execute(query)
        if self.cursor.description is None:
            raise ValueError("The SQL query did not return rows")
        return self.cursor.fetchall()


    def close(self):
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None
