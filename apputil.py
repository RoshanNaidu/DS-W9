import pandas as pd

class GroupEstimate(object):
    def __init__(self, estimate='median'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_values = None
    
    def fit(self, X, y):
        df = pd.concat([X, y.rename('y')], axis=1)
        if self.estimate == 'mean':
            self.group_values = df.groupby(list(X.columns))['y'].mean()
        else:
            self.group_values = df.groupby(list(X.columns))['y'].median()

    def predict(self, X_list):
        preds, missing = [], 0
        for row in X_list:                      
            key = tuple(row)
            if key in self.group_values.index:
                preds.append(self.group_values.loc[key])
            else:
                preds.append(np.nan)
                missing += 1
        if missing:
            print(f"{missing} group(s) missing from training data.")
        return np.array(preds)
