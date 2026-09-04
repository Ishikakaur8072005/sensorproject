from pymongo.mongo_client import MongoClient
import pandas as pd
import json
import os
from src.constant import MONGO_DB_URL, MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME

try:
    client = MongoClient(MONGO_DB_URL, serverSelectionTimeoutMS=5000)
    csv_path = os.path.join("notebooks", "wafer_secom_cleaned.csv")
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    json_record = list(json.loads(df.T.to_json()).values())

    client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME].drop()
    client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME].insert_many(json_record)
    print(f"Successfully uploaded {len(json_record)} records from {csv_path} to MongoDB.")
except Exception as e:
    print(f"MongoDB upload skipped due to connection timeout ({e}). Data Ingestion fallback will use local notebooks/wafer_secom_cleaned.csv.")