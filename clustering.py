import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from kmodes.kprototypes import KPrototypes
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
from matplotlib.text import Annotation


# Edit the text on the tree node
def edit_node(node):
    if type(node) == Annotation:
        txt = node.get_text()
        txt = re.sub(" <=", "\n<=", txt)
        txt = re.sub("\d+\n\\[.*\\]\n", "", txt)
        node.set_text(txt)
    return node


# Train a decision tree
def train_decision_tree(features, target, num_clusters, max_depth):
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.8,
                                                                                random_state=1)
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree = decision_tree.fit(features_train, target_train)
    test_acc = accuracy_score(target_test, decision_tree.predict(features_test))
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(decision_tree, feature_names=dataset_encoded.columns,
              class_names=["Clus " + str(j) for j in range(1, num_clusters + 1)],
              filled=True,
              impurity=False,
              label=False,
              ax=ax,
              fontsize=12)
    ax.properties()['children'] = [edit_node(i) for i in ax.properties()['children']]
    fig.show()
    return decision_tree, test_acc


if __name__ == '__main__':
    # Load the data set
    dataset = pd.read_csv("marketing_campaign.csv", sep='\t')

    # Clean the data set
    dataset = dataset.dropna().reset_index()
    dataset = dataset[dataset['Income'] != 666666].reset_index()

    # Feature engineering
    dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'], format="%d-%m-%Y").apply(lambda x: x.timestamp())
    numerical = ['Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    categorical = ['Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                   'AcceptedCmp5', 'Complain', 'Response']
    binary_categorical = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain',
                          'Response']
    nominal_categorical = ['Marital_Status']
    ordinal_categorical = ['Education']
    dataset = dataset[numerical + categorical]

    # Categorical feature encoding using one-hot encoding
    one_hot = OneHotEncoder()
    one_hot = one_hot.fit(dataset[nominal_categorical])
    nominal_encoded = pd.DataFrame(one_hot.transform(dataset[nominal_categorical]).toarray(),
                                   columns=one_hot.get_feature_names_out())

    # Categorical feature encoding using integer encoding
    int_encoder = LabelEncoder()
    int_encoder = int_encoder.fit(['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'])
    ordinal_encoded = pd.DataFrame(int_encoder.transform(dataset[ordinal_categorical]), columns=ordinal_categorical)

    # Standardization on the numerical features
    scalar = StandardScaler()
    scalar.fit(dataset[numerical])
    dataset_norm_numerical = pd.DataFrame(scalar.transform(dataset[numerical]), columns=numerical)

    # Summary data set
    dataset_encoded = dataset[numerical].join(dataset[binary_categorical].join(nominal_encoded.join(ordinal_encoded)))
    dataset_encoded_norm = \
        dataset_norm_numerical.join(dataset[binary_categorical].join(nominal_encoded.join(ordinal_encoded)))
    dataset_norm = dataset_norm_numerical.join(dataset[categorical])

    # Clustering
    run_k_prototypes = True
    run_hierarchical = True

    # K-prototypes clustering
    if run_k_prototypes:
        clusters_SSEs = []
        clusters = []
        ks = range(2, 21)
        print('Running K-prototypes clustering...')
        for k in ks:
            print("\tRunning K-prototypes with K = " + str(k))
            kproto = KPrototypes(n_clusters=k)
            kproto.fit(dataset_norm,
                       categorical=[dataset_norm.columns.tolist().index(col_name) for col_name in categorical])
            clusters.append(kproto)
            clusters_SSEs.append(kproto.cost_)

        # Elbow method
        kneedle = KneeLocator(ks, clusters_SSEs, S=1.0, curve="convex", direction="decreasing")
        optimal_k = round(kneedle.elbow)
        kneedle.plot_knee()
        plt.show()
        print("\nThe optimal number of clusters K = " + str(optimal_k))
        optimal_clustering = clusters[optimal_k - 2]

        # Show the centroids of the optimal clustering
        print('\nThe centroids found by the K-means clustering are:')
        print(pd.DataFrame(optimal_clustering.cluster_centroids_,
                           index=['Centroids ' + str(i) for i in range(1, optimal_k + 1)],
                           columns=dataset_norm.columns))
        clustering_result = optimal_clustering.predict(dataset_norm,
                                                       categorical=[dataset_norm.columns.tolist().index(col_name)
                                                                    for col_name in categorical])
        result = dataset.copy(True)
        result['K-prototypes Clustering'] = clustering_result
        print('\nClustering Result:')
        print(result)
        print("\nInterpret the clustering using decision tree...")
        decision_tree, test_acc = train_decision_tree(dataset_encoded, clustering_result, optimal_k, 4)
        print("\tThe accuracy of the decision tree is " + str(test_acc))
        print('\n=========================================================================')

    # Hierarchical clustering
    if run_hierarchical:
        print('\n\nRunning Hierarchical clustering...')
        clustering = sch.linkage(dataset_encoded_norm, method='ward')
        sch.dendrogram(clustering)
        plt.show()

        # Elbow method
        num_clusters = range(2, 21)
        clusters_SSEs = []
        for k in num_clusters:
            label = sch.fcluster(clustering, k, 'maxclust')
            clusters_SSE = 0
            for i in range(1, k + 1):
                centroid = dataset_encoded_norm[pd.Series(label) == i].mean()
                clusters_SSE += np.square(dataset_encoded_norm[pd.Series(label) == i] - centroid).sum().sum()
            clusters_SSEs.append(clusters_SSE)
        kneedle = KneeLocator(num_clusters, clusters_SSEs, S=1.0, curve="convex", direction="decreasing")
        optimal_num_clusters = round(kneedle.elbow)
        kneedle.plot_knee()
        plt.show()

        print("\nThe optimal number of clusters is " + str(optimal_num_clusters))
        clustering_result = sch.fcluster(clustering, optimal_num_clusters, 'maxclust')
        result = dataset.copy(True)
        result['Hierarchical Clustering'] = clustering_result
        print('\nClustering Result:')
        print(result)
        print("\nInterpret the clustering using decision tree...")
        decision_tree, test_acc = train_decision_tree(dataset_encoded, clustering_result, optimal_num_clusters, 4)
        print("\tThe accuracy of the decision tree is " + str(test_acc))
        print('\n=========================================================================')
