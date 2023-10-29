import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
import shutil
from joblib import Memory
import joblib
from sklearn.cluster import DBSCAN
from scipy.cluster import  hierarchy
from sklearn.cluster import KMeans

# Sử dụng joblib để đệm kết quả
memory = Memory("cachedir", verbose=0)

@memory.cache
def hierarchical_clustering(vectors):
    clusterer = KMeans(n_clusters=10, random_state=0, n_init="auto")
    
    clusterer.fit(vectors)
    labels = clusterer.labels_
    print(labels)
    print(set(labels))
    joblib.dump(clusterer, "clustering.pt")

    return labels
data_folder = '/mlcv/WorkingSpace/Personals/ngocnd/BKAI2023/BKAI-NAVER-OCR-2023/Data/Backbone_vector/DataTrain'
file_paths = sorted([os.path.join(data_folder, file) for file in os.listdir(data_folder)], key=lambda x: int(os.path.basename(x).split(".")[0]))

vectors = []
for idx, file in enumerate(file_paths):
    if idx > 1000:
        break
    vectors.append(np.load(file).flatten())
    if (idx+1) % 10 == 0:  
        print(f"Loaded {idx+1} vectors...")


vectors = np.array(vectors)

with open('/mlcv/WorkingSpace/Personals/ngocnd/BKAI2023/BKAI-NAVER-OCR-2023/Data/Backbone_vector/vector.txt', 'r') as f:
    image_paths = [path.strip() for path in f.readlines()][:1000]
print('Load image done')

labels = hierarchical_clustering(vectors)
print('Train done')

 
