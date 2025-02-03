import mlx
import metal

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return mlx.array(data, dtype=np.float32)

def encode_sentences(sentences, embedding_model):
    embeddings = []
    for sentence in sentences:
        embeddings.append(embedding_model.encode(sentence))
    return mlx.array(embeddings, dtype=np.float32)

def create_matrix(rows, columns):
    return mlx.random.rand(rows, columns).astype(np.float32)
