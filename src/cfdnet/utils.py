import torch

def save_model(model, filepath):
    """Saves the model weights."""
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath, *model_args):
    """Loads model weights into a model."""
    model = model_class(*model_args)
    model.load_state_dict(torch.load(filepath))
    return model

def generate_positional_encodings(seq_len, d_model):
    """Generates sinusoidal positional encodings."""
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def visualize_embeddings(embeddings, labels=None):
    """Visualizes embeddings in 2D using PCA or t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    reduced_embeddings = PCA(n_components=2).fit_transform(embeddings.cpu().detach().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels if labels is not None else 'b', cmap='rainbow')
    plt.colorbar()
    plt.title('Embedding Visualization')
    plt.show()
