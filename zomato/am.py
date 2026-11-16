import snowflake.connector
import getpass

password = getpass.getpass("Snowflake password: ")

try:
    conn = snowflake.connector.connect(
        account="BHMEXZT-FV35063",
        user="AMITHVARDHANGD3@GMAIL.COM",
        password=password
    )

    print("SUCCESS: Connected using login_name!")
    conn.close()

except Exception as e:
    print("FAILED:", e)