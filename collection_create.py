from pymilvus import MilvusClient, connections, Collection


connections.connect("default", host="localhost", port="19530")
client = MilvusClient("voices.db")
client.drop_collection(collection_name="Voice_Collection")
client.create_collection(
    collection_name="Voice_Collection",
    dimension=390
)

res = client.get_load_state(
    collection_name="Voice_Collection"
)

print(res)