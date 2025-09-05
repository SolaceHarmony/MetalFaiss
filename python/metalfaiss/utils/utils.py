import mlx
import metal

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return mlx.core.eval.array(data, dtype=np.float32)

def encode_sentences(sentences, embedding_model):
    embeddings = []
    for sentence in sentences:
        embeddings.append(embedding_model.encode(sentence))
    return mlx.core.eval.array(embeddings, dtype=np.float32)

def create_matrix(rows, columns):
    return mlx.core.eval.random.rand(rows, columns).astype(np.float32)

def normalize_data(data):
    data = mlx.core.eval.array(data, dtype=np.float32)
    norms = mlx.core.eval.linalg.norm(data, axis=1, keepdims=True)
    return data / norms

def compute_distances(data, query):
    data = mlx.core.eval.array(data, dtype=np.float32)
    query = mlx.core.eval.array(query, dtype=np.float32)
    return mlx.core.eval.linalg.norm(data - query, axis=1)

def random_projection(data, output_dim):
    data = mlx.core.eval.array(data, dtype=np.float32)
    projection_matrix = mlx.core.eval.random.randn(data.shape[1], output_dim).astype(np.float32)
    return mlx.core.eval.dot(data, projection_matrix)
