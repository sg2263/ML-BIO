import pandas as pd
from sklearn.feature_selection import VarianceThreshold

csv_file_path = 'your_dataset.csv'
data = pd.read_csv(csv_file_path)


X = data.drop('target_column_name', axis=1)
y = data['target_column_name']


threshold_value = 0.1


variance_selector = VarianceThreshold(threshold=threshold_value)


X_high_variance = variance_selector.fit_transform(X)

selected_features = X.columns[variance_selector.get_support()]

# Print now
print("Selected Features:")
print(selected_features)


X_selected = pd.DataFrame(X_high_variance, columns=selected_features)

