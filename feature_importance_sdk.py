import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

class FeatureImportanceSDK:
    def __init__(self, model):
        if not hasattr(model, 'fit'):
            raise ValueError("The provided model does not have a 'fit' method.")
        self.model = model

    def calculate_importance(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y should be numpy arrays.")
        
        if X.shape[1] > 20:
            raise ValueError("The number of features exceeds the limit of 20.")
    
        try:
            self.model.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"An error occurred while fitting the model: {e}")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("The provided model does not have 'feature_importances_' attribute.")
        
        self.importances = self.model.feature_importances_
        return self.importances

    def plot_importance(self, feature_names, color='blue'):
        if not hasattr(self, 'importances'):
            raise RuntimeError("Feature importances have not been calculated. Call 'calculate_importance' first.")
        
        if len(feature_names) != len(self.importances):
            raise ValueError("The length of feature_names does not match the number of features.")
        
        indices = np.argsort(self.importances)[::-1]
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(len(self.importances)), self.importances[indices], align="center", color=color)
        plt.xticks(range(len(self.importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names

    model = RandomForestClassifier()
    sdk = FeatureImportanceSDK(model)
    importances = sdk.calculate_importance(X, y)
    print("Feature Importances:", importances)
    
    # Plot with default color
    sdk.plot_importance(feature_names)
    
    # Plot with custom color
    sdk.plot_importance(feature_names, color='green')