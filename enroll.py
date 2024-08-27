import record
import process
import numpy as np
import json
import pandas as pd
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

URI=os.getenv("MONGO_URL")
mongodb_client = MongoClient(URI)
database = mongodb_client["VoiceRec"]
collection = database["voices"]
name=input("Enter Name To Enroll: ")

file_name=(name+"/"+name+"0")
record.record_audio(name+str(0))
vectors=process.process_file(file_name)
print(vectors.shape)
print(len(vectors.tolist()))
print(len(vectors.tolist()[0]))
obj={"matrix":vectors.tolist(), "name":name}
collection.insert_one(obj)
mongodb_client.close()






