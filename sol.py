# Import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# Loading and examining the dataset
penguins_df = pd.read_csv("data/penguins.csv")
#print(len(penguins_df), "Initial reading in")


penguins_clean = penguins_df.dropna()
penguins_clean[penguins_clean['flipper_length_mm']>4000]
penguins_clean[penguins_clean['flipper_length_mm']<0]
penguins_clean = penguins_clean.drop([9,14])




penguins_clean = penguins_df.dropna()
#print(len(penguins_clean), "After dropping NA")


penguins_clean[penguins_clean['flipper_length_mm']>4000]
penguins_clean[penguins_clean['flipper_length_mm']<0]
penguins_clean = penguins_clean.drop([9,14])


#print(penguins_clean)
df = pd.get_dummies(penguins_clean).drop('sex_.',axis=1)
penguins_df.boxplot()  
plt.show()
scaler = StandardScaler()
penguins_preprocessed = scaler.fit_transform(df)

# See components that explain more than 10% of the data
pca = PCA()
penguins_preprocessed_pca = pca.fit_transform(penguins_preprocessed)
cumulative_r = 0
n_components=0
for component in pca.explained_variance_ratio_:
    if component > 0.1:
        n_components = n_components + 1 
    cumulative_r += component

print(cumulative_r, n_components)

pca = PCA(n_components)
penguins_preprocessed_pca = pca.fit_transform(penguins_preprocessed)

kmeans_inertias = list()
# Create for loop
for k in np.arange(1,10,step=1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(penguins_preprocessed_pca)
    kmeans_inertias.append(kmeans.inertia_)
n_clusters = 4


plt.plot(np.arange(1,10,step=1),kmeans_inertias , 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()





kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(penguins_preprocessed_pca)
plt.scatter(penguins_preprocessed_pca[:,0] , penguins_preprocessed_pca[:,1], c=kmeans.labels_)
penguins_clean.insert(5, 'label', kmeans.labels_ ,True)
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','body_mass_g']

stat_penguins = penguins_clean.groupby('label')[numeric_columns].mean()
