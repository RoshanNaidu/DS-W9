import pandas as pd

class GroupEstimate(object):
    def __init__(self, estimate):
        self.estimate = estimate
    
    def fit(self, X, y):
        # Combine X and y into a single DataFrame
        data = pd.DataFrame(X).copy()
        data['target'] = y
        
        # Calculate the group estimate based on the specified method
        if self.estimate == "mean":
            self.group_estimates = data.groupby(self.estimate)['target'].mean().to_dict()
        elif self.estimate == "median":
            self.group_estimates = data.groupby(self.estimate)['target'].median().to_dict()
        else:
            raise ValueError("Estimate must be either 'mean' or 'median'")
        
        return self

    def predict(self, X):
        # Predict the estimate based on the group estimates
        predictions = []
        missing_count = 0
        for value in X[self.estimate]:
            if value in self.group_estimates:
                predictions.append(self.group_estimates[value])
            else:
                predictions.append(float('nan'))  # or some default value
                missing_count += 1
        if missing_count > 0:
            print(f"Warning: {missing_count} missing groups were found during prediction.")
        return predictions
    
