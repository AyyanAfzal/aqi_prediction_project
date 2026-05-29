import os
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

load_dotenv(".env")

host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")

if not host:
    raise ValueError("DATABRICKS_HOST is missing from .env")

if not token:
    raise ValueError("DATABRICKS_TOKEN is missing from .env")

w = WorkspaceClient(
    host=host,
    token=token
)

current_user = w.current_user.me()

print("Connected to Databricks successfully.")
print("User:", current_user.user_name)