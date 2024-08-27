import record
import process
from numpy.linalg import norm
import numpy as np
import pandas as pd
from pymilvus import MilvusClient, connections, Collection
from pymongo import MongoClient
from scipy.spatial.distance import euclidean

URI="mongodb+srv://kundurthisumedha:K2WQp4meOd8WFY5z@cluster0.pspbr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_client = MongoClient(URI)
database = mongodb_client["VoiceRec"]
collection = database["voices"]


def pad_matrices(matrix1, matrix2):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    max_rows = max(rows1, rows2)
    padded_matrix1 = np.pad(matrix1, ((0, max_rows - rows1), (0, 0)), mode='constant')
    padded_matrix2 = np.pad(matrix2, ((0, max_rows - rows2), (0, 0)), mode='constant')

    return padded_matrix1, padded_matrix2

def calculate_euclidean_distance(matrix1, matrix2):
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()
    distance = euclidean(flattened_matrix1, flattened_matrix2)
    return distance

def search(matrix):
    min_distance=float('inf')
    name=""
    documents = collection.find()
    for document in documents:
        array = np.array(document["matrix"])
        padded_matrix1,padded_matrix2=pad_matrices(matrix,array)
        distance = calculate_euclidean_distance(padded_matrix1,padded_matrix2)
        if distance<min_distance:
            min_distance=distance
            name=document["name"]   
    return name