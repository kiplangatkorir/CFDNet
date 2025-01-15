# CFDNet - Continuous Function Decomposition Network

CFDNet is a deep learning model designed for sequential tasks such as text processing, time-series prediction, and other tasks requiring sequential data modeling. It leverages continuous function decomposition with spline-based positional embeddings and decomposition blocks for powerful feature extraction and transformation.

## Features

- **Spline-Based Positional Embeddings**: Uses learnable spline functions for continuous positional encoding, replacing traditional methods like sinusoidal embeddings.
- **Decomposition Blocks**: Introduces the concept of univariate transformations with learnable mixing, which provides flexible feature extraction.
- **Residual Connections**: Ensures stable learning by using residual connections between decomposition blocks.
- **Flexible Architecture**: The model is modular, making it suitable for various applications like text classification, sequence generation, and more.

## Installation

You can install CFDNet by cloning the repository and using `pip` to install it locally or from a remote source.

### Install from GitHub (Recommended)
To install directly from GitHub:

```bash
pip install git+https://github.com/yourusername/cfdnet.git
```

### Install via Local Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cfdnet.git
   cd cfdnet
   ```

2. Install the package locally:

   ```bash
   pip install .
   ```

### Dependencies

- `torch` (PyTorch) - Deep learning framework
- `numpy` - Numerical operations
- `matplotlib` (Optional) - For visualization (e.g., embedding visualizations)
- `scikit-learn` (Optional) - For PCA or t-SNE visualizations

## Usage

Once installed, you can use the `CFDNet` class for various tasks like text classification, time-series prediction, etc.

### Example: Text Classification

```python
import torch
from cfdnet.models import CFDNet

# Model configuration
vocab_size = 5000  # Number of words in vocabulary
d_model = 128  # Dimensionality of the model's embeddings
num_layers = 6  # Number of decomposition blocks
max_seq_len = 256  # Maximum length of input sequences

# Create the CFDNet model
model = CFDNet(vocab_size, d_model, num_layers, max_seq_len)

# Example input sequence (batch_size, seq_len)
input_seq = torch.randint(0, vocab_size, (2, 256))  # Randomly generated sequence

# Forward pass
output = model(input_seq)
print(output.size())  # Expected output: torch.Size([2, 256, 5000])
```

### Example: Time-Series Prediction

For time-series data, you can use CFDNet to predict future values based on past observations. This example assumes you're working with univariate time-series data.

```python
import torch
from cfdnet.models import CFDNet

# Model configuration
vocab_size = 100  # Number of unique time-series values
d_model = 64  # Dimensionality of the model's embeddings
num_layers = 4  # Number of decomposition blocks
max_seq_len = 100  # Maximum length of input sequences

# Create the CFDNet model
model = CFDNet(vocab_size, d_model, num_layers, max_seq_len)

# Example time-series input (batch_size, seq_len)
input_seq = torch.randint(0, vocab_size, (2, 100))  # Randomly generated time-series

# Forward pass
output = model(input_seq)
print(output.size())  # Expected output: torch.Size([2, 100, 100])
```

### Save and Load Model

You can easily save and load the trained model.

#### Save Model:

```python
import torch
from cfdnet.utils import save_model

# Save the model
save_model(model, 'cfdnet_model.pth')
```

#### Load Model:

```python
from cfdnet.utils import load_model
from cfdnet.models import CFDNet

# Load the model
loaded_model = load_model(CFDNet, 'cfdnet_model.pth', vocab_size, d_model, num_layers, max_seq_len)
```

### Visualization (Optional)

You can visualize embeddings using PCA or t-SNE.

```python
from cfdnet.utils import visualize_embeddings
import torch

# Generate random embeddings (e.g., output of the model)
embeddings = torch.randn(100, 64)  # 100 samples, 64-dimensional embeddings

# Visualize embeddings (requires matplotlib and scikit-learn)
visualize_embeddings(embeddings)
```

## Architecture Overview

### Spline-Based Positional Embeddings

CFDNet uses **learnable spline functions** to generate continuous positional embeddings. This method allows the model to learn more flexible positional encodings that can adapt to different sequence lengths and patterns.

### Decomposition Blocks

The core of CFDNet consists of **decomposition blocks**, which apply univariate transformations and combine them via learnable mixing layers. This allows the model to extract diverse features at each layer, with each block focusing on different aspects of the input.

### Residual Connections

To facilitate stable training, residual connections are used in between the decomposition blocks. This helps prevent vanishing gradients and accelerates learning, especially for deeper models.

## Examples

### Text Classification

Use CFDNet to perform text classification tasks. Simply feed sequences of tokenized words into the model, and train it on your labeled dataset.

### Time-Series Prediction

Use CFDNet to forecast future values based on historical time-series data. Adjust the input sequence length (`max_seq_len`) and the model's output to suit your specific prediction task.

### Other Applications

CFDNet is flexible and can be applied to other sequential tasks, such as sequence-to-sequence generation, anomaly detection in time-series, and more.

## Testing

To test the functionality of the library, run the test suite:

```bash
pytest tests
```

This will execute the tests and provide feedback on whether everything is working as expected.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions to CFDNet! If you have an idea or bug fix, feel free to open an issue or create a pull request.

- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Commit your changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request to the main repository.
