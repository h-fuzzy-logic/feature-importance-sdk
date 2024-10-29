
- `feature_importance_sdk.py`: The main SDK implementation.
- `examples/example.ipynb`: A Jupyter Notebook demonstrating how to use the SDK.
- `requirements.txt`: A list of dependencies required to run the SDK and examples.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- `pip` (Python package installer)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/h-fuzzy-logic/feature-importance-sdk.git
    cd feature-importance-sdk
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

### Usage

#### Example Script

You can use the `FeatureImportanceSDK` in your Python scripts. Below is an example usage:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from feature_importance_sdk import FeatureImportanceSDK

# Load data
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Initialize model
model = RandomForestClassifier()

# Use FeatureImportanceSDK
sdk = FeatureImportanceSDK(model)
importances = sdk.calculate_importance(X, y)
print("Feature Importances:", importances)

# Plot with default color
sdk.plot_importance(feature_names)

# Plot with custom color
sdk.plot_importance(feature_names, color='orange')