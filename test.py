import record
import process
from numpy.linalg import norm
import numpy as np
import pandas as pd
from pymilvus import MilvusClient, connections, Collection


connections.connect("default", host="localhost", port="19530")
client = MilvusClient("voices.db")


def search(vector):
    print("FLATTENED VECTOR")
    print(vector)
    res = client.search(
    collection_name="Voice_Collection",  # target collection
    data=[vector],  # query vectors
    limit=10,  # number of returned entities
    output_fields=["name"],  # specifies fields to be returned
    )
    return res






# speaker=get_speaker(vectors)
# print(speaker)
# # ask user if correct, or if not, who?
# ans=input("Is it correct (Y/N)?")
# if ans=="Y":
#     result = df[df['Name'] == speaker]
#     current_vectors=result.iloc[0]['Voice-id']
#     row_index = df[df['Name'] == speaker].index[0]
#     v=[current_vectors]
#     v.append(vectors)
#     mean_v=np.mean(v, axis=0)
#     print(mean_v)
#     df.at[row_index, 'Voice-id'] = mean_v


# elif ans=="N":
#     ans=input("Correct Ans? ")
#     result = df[df['Name'] == ans]
#     current_vectors=result.iloc[0]['Voice-id']
#     row_index = df[df['Name'] == ans].index[0]
#     v=[current_vectors]
#     v.append(vectors)
#     mean_v=np.mean(v, axis=0)
#     print(mean_v)
#     df.at[row_index, 'Voice-id'] = mean_v
# df.to_json("voices copy.json")
