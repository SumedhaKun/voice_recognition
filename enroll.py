import record
import process
import numpy as np
import json
import pandas as pd
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, Collection, connections, utility

connections.connect("default", host="localhost", port="19530")
client = MilvusClient("voices.db")
colls=client.list_collections()
print(colls)
name=input("Enter Name To Enroll: ")
file_names=[]
for i in range(1):
    file_names.append(name+"/"+name+str(i))
    record.record_audio(name+str(i))
vectors=process.process_files(file_names)
flat=vectors[0].flatten()[0:390]
print(flat)
i=client.get_collection_stats("Voice_Collection")["row_count"]
data = [
    {"id": i, "vector": flat, "name": name}
]

res = client.insert(collection_name="Voice_Collection", data=data)
print(res)






