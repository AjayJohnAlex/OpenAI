from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("C:\\Users\\ajayj\\OneDrive\\Desktop\\HEA enthalpy.xlsx")


df.head()

# Extract numerical feature (Enthalpy values)
X = df[['Enthalpy (kJ/mol)']]  # DataFrame with only the 'Enthalpy' column



# Standardize the features (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize empty list to store inertia values
inertia = []

# Calculate inertia for different values of K
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), inertia, marker='o', linestyle='--')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid()
plt.show()


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Initialize KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)

# Fit the KMeans model
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Print DataFrame with cluster labels
print("DataFrame with Cluster Labels:")
print(df)

# Print centroids
print("Centroids:")
for j, centroid in enumerate(centroids):
    print(f"Cluster {j}: Centroid = {centroid[0]} kJ/mol")
    
    


# -------------------------------------------------------------------
# -------------------------------------------------------------------

# New data point
new_data_point = [[-22]]

# Scale the new data point using the same scaler
new_data_point_scaled = scaler.transform(new_data_point)

# Predict cluster for the new data point
predicted_cluster = kmeans.predict(new_data_point_scaled)[0]

print(f"Predicted cluster for new data point (Enthalpy = {new_data_point}): {predicted_cluster}")


df[df['Cluster']==1]


df.to_csv("clustered_HEA.csv", index=False)




