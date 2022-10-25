

import sys
print(sys.version)



# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams


import sklearn
import sklearn.metrics as sm

# %%
from sklearn.cluster import AgglomerativeClustering

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# %%
np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10, 3))
# %matplotlib inline
plt.style.use('seaborn-whitegrid')

# %%
# !pwd

# %%
address = '/home/alois/Documents/development/python_data_science_essential_training_part_2//Data/mtcars.csv'

cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

X = cars[['mpg', 'disp', 'hp', 'wt']].sort_values

y = cars.iloc[:,(9)].values

# %% [markdown]
# ### Using scipy to generate dendrograms

# %%
Z = linkage(X, 'ward')

# %%
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15, show_contracted=True)

plt.title('Truncated Hierarchial Clustering Diagram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()

# %% [markdown]
# ## Generating hierarchical clusters

# %%
k=2

Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)

# %%
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)

# %%
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')
Hclustering.fit(X)

sm.accuracy_score(y, Hclustering.labels_)

# %%



