import numpy as np
import metal

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return np.array(data, dtype=np.float32)

def encode_sentences(sentences, embedding_model):
    embeddings = []
    for sentence in sentences:
        embeddings.append(embedding_model.encode(sentence))
    return np.array(embeddings, dtype=np.float32)

def create_matrix(rows, columns):
    return np.random.rand(rows, columns).astype(np.float32)
