import pandas as pd


def csv_read(filepath, encoding="utf-8"):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table
    """
    return pd.read_csv(filepath, encoding=encoding)


def csv_write(df, filepath, index=False, encoding="utf-8"):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table
    """
    return df.to_csv(filepath, index=index, encoding=encoding)


def xlsx_read(filepath, sheet_name):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-excel
    """
    return pd.read_excel(filepath, sheet_name, index_col=None, na_values=["NA"])


def xlsx_write(df, filepath, sheet_name):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-excel
    """
    return df.to_excel(filepath, sheet_name=sheet_name)


def mysql_read(sql, conn):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-sql
    :param conn:
        ```python
        import pymysql

        db_host = "host"
        db_port = 3306
        db_user = "user name"
        db_pass = "pass word"
        db_dbnm = "database"

        from sqlalchemy import create_engine
        link = "%s:%s@%s:%d/%s?charset=utf8" % (db_user, db_pass, db_host, db_port, db_dbnm)
        conn = create_engine("mysql+pymysql://" + link, encoding="utf8")
        ```
    """
    return pd.read_sql(sql, conn)


def mysql_write(df, table_name, conn, if_exists="replace", index=False, chunksize=1000):
    """
    http://pandas.pydata.org/pandas-docs/stable/io.html#io-sql
    :param conn:
        ```python
        import pymysql

        db_host = "host"
        db_port = 3306
        db_user = "user name"
        db_pass = "pass word"
        db_dbnm = "database"

        from sqlalchemy import create_engine
        link = "%s:%s@%s:%d/%s?charset=utf8" % (db_user, db_pass, db_host, db_port, db_dbnm)
        conn = create_engine("mysql+pymysql://" + link, encoding="utf8")
        ```
    """
    return df.to_sql(table_name, conn, if_exists=if_exists, index=index, chunksize=chunksize)