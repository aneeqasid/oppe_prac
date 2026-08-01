from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load standard Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target

# Map numbers to names
species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
df['species'] = df['species'].map(species_map)

# Split into v0 and v1 (75 rows each)
df_v0 = df.iloc[:75].copy()
df_v1 = df.iloc[75:].copy()

# Add a few NaN values in features for imputation practice
np.random.seed(42)
for col in ['sepal_length', 'sepal_width']:
    df_v0.loc[np.random.choice(df_v0.index, size=5, replace=False), col] = np.nan
    df_v1.loc[np.random.choice(df_v1.index, size=5, replace=False), col] = np.nan

# Save to data folder
df_v0.to_csv('data/iris_v0.csv', index=False)
df_v1.to_csv('data/iris_v1.csv', index=False)

print("✅ Generated data/iris_v0.csv and data/iris_v1.csv with 75 rows each.")
print(f"v0 shape: {df_v0.shape}")
print(f"v1 shape: {df_v1.shape}")
