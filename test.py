from numpy.linalg import norm
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neural_network import MLPClassifier
LE = LabelEncoder()

URI="mongodb+srv://kundurthisumedha:K2WQp4meOd8WFY5z@cluster0.pspbr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_client = MongoClient(URI)
database = mongodb_client["VoiceRec"]
collection = database["voices"]
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the GRU
        return out

def calculate_euclidean_distance(matrix1, matrix2):
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()
    distance = euclidean(flattened_matrix1, flattened_matrix2)
    return distance

def search(matrix):
    # random forest
    df = pd.DataFrame(list(collection.find()))

    df["matrix"]=[np.array(l) for l in df["matrix"]]
    df['name_id'] = LE.fit_transform(df['name'])

    labels=df["name_id"]
    X = np.array(df['matrix'].tolist())
    y=labels.values

    # knn:
    model= KNeighborsClassifier()
    clf =model.fit(X, y)
    pred=clf.predict([matrix.flatten().tolist()])
    print((LE.inverse_transform([pred]))[0])

    # random forest:
    model= RandomForestClassifier()
    clf =model.fit(X, y)
    print("Random Forest")
    pred=clf.predict([matrix.flatten().tolist()])
    print((LE.inverse_transform([pred]))[0])

    # svm:
    clf = svm.SVC()
    clf.fit(X, y)
    print("SVM")
    pred=clf.predict([matrix.flatten().tolist()])
    print((LE.inverse_transform([pred]))[0])

    #nn:
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(9, 5), random_state=1)
    clf.fit(X, y)
    print("MLP")
    pred=clf.predict([matrix.flatten().tolist()])
    print((LE.inverse_transform([pred]))[0])
    return (LE.inverse_transform([pred]))[0]