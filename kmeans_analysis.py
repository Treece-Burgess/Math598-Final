#!/usr/bin/env python3

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from kneed import KneeLocator


class KmeansAnalysis:
    def __init__(self, feature=None, target=None, normalization = "standard"):
        """
        Arguments
        ----------
            feature (ndarray): 
        
            target (ndarray): 
            
            normalization (str): Optional argument, default is "standardization"
                                 Data scaling, with three available options:
                                 - "standardization" = StandardScaler()
                                 - "minmax" = MinMaxScaler()
                                 - "robust" = RobustScaler()
        """
        self.feature = feature
        self.target = target
        
        if not isinstance(self.feature, np.ndarray):
            raise ValueError("Must pass in a numpy array for the argument feature.")
        if not isinstance(self.target, np.ndarray):
            raise ValueError("Must pass in a numpy array for the argument target.")
        
        if normalization == "standard":
            self.normalization = StandardScaler()
        elif normalization == "minmax":
            self.normalization = MinMaxScaler()
        elif normalization =="robust":
            self.normalization = RobustScaler()
        else: 
            raise ValueError("Please enter a valid normalization technique: standardization, minmax, or robust")
        
        self.scaler = self.normalization
        self.scaled_data=self.scaler.fit_transform(self.feature)
        
    def cluster(self, **kmeans_kwargs):
        # performing feature scaling
        set_normalization = self.normalization
        scaled_feature = set_normalization.fit_transform(self.feature)
        
        # finding optimal number of clusters
        sil_coeff = []
        for s in range(2, 11):
            kmeans = KMeans(n_clusters=s, **kmeans_kwargs)
            kmeans.fit_predict(scaled_feature)
            sil_coeff.append(silhouette_score(scaled_feature, kmeans.labels_))
        # defining number of clusters k     
        k = sil_coeff.index(max(sil_coeff)) + 2
        
        
        opt_kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        
        prediction = opt_kmeans.fit_predict(scaled_feature)
    
        
        centroids = opt_kmeans.cluster_centers_
        # plot clustering results
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 7)
        ax.scatter(scaled_feature[:, 0], scaled_feature[:, 1], c=prediction, cmap = "viridis")
        ax.scatter(centroids[:,0], centroids[:,1] , s = 300, c='r',marker="H")
        # Plot cluster centers
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="H",
            c="red",
            alpha=1,
            s=600,
            edgecolor="k",
        )
        # Label cluster centers on plot
        for i, c in enumerate(centroids):
                ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Clustering Plot for Featured Data")
        
        # Generate actual clustering labels
        label_encoder = LabelEncoder()
        true_labels = label_encoder.fit_transform(self.target)
    
        
        labels = kmeans.labels_ #ask Ian why we get values of 8 and such
        
        #may wany to add check to see if they already passs true labels
                
        correct_labels = sum(true_labels == labels)

        print("Result: %d out of %d samples were correctly labeled." % (correct_labels, self.target.size))
        print('Total Cluster Accuracy: {0:0.2f}'.format(correct_labels/float(self.target.size)))

        
        return ax
    
    def elbow_method(self, **kmeans_kwargs):
        """Displays an elbow plot to show optimal number of clusters.
        
        Arguments
        ----------
        kmeans_kwargs (dict): Set options for Kmeans algorithm parameters
        
        Returns
        ----------
        plot (axes): Elbow plot to be displayed to user. A redline will be marked through the optimal
                     number of clusters.
        """
        
        # calculate optimal cluster
        sse = []
        for k in range(2,11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(self.scaled_data)
            sse.append(kmeans.inertia_)
        
        print(sse)
        kl = KneeLocator(
            range(2, 11), sse, curve="convex", direction="decreasing"
        )
        
        # plotting elbow plot
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 7)
        plt.style.use("seaborn-whitegrid")
        ax.axvline(kl.elbow, ls='--', color='r')
        ax.plot(range(2, 11), sse, linestyle='--', marker="o", color="b", markersize=10)
        ax.set_xlabel("k-number of clusters")
        ax.set_ylabel("wcss")
        plt.xticks(range(2,11))
        plt.title("Elbow Plot")
    
    
    def dunn_index(self):
        # Will need to think on this more
        print(1)
        
    def inertia(self, **kmeans_kwargs):
        """Display a bar chart to show magnitude of inertia values depending on number of clusters.
        
         Arguments
        ----------
        kmeans_kwargs (dict): Set options for Kmeans algorithm parameters
        
        Returns
        ----------
        plot (axes): Bar plot to be displayed to user.
        """
        inertia = []
        for k in range(2,11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
    
        fig, ax = plt.subplots()
        plt.style.use("seaborn-whitegrid")
        ax.bar(range(2, 11),inertia, color="b")
        ax.set_xticks(list(range(2,11)))
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia Value")
        ax.set_title("Bar Chart: Inertia Scores")
        
    def silhouette(self, num_clusters=None, show_clustering = False, **kmeans_kwargs):
        """Displays silhouette and clustering plot.
        
        Arguments
        ----------
        num_clusters (num): Number of clusters to test via silhouette plot.
                            Number of clusters to test will be k=2 to k=num_clusters.
                            
        kmeans_kwargs (dict): Set options for Kmeans algorithm parameters
        
        Returns
        ----------
        plot (axes): Silhouette chart and clustering plot.    
        """
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax1.set_ylim([0, len(self.feature) + (num_clusters + 1) * 10])
            
        clusterer = KMeans(n_clusters=num_clusters, n_init=10, random_state=10)
        cluster_labels = clusterer.fit_predict(self.feature)
            
        silhouette_avg = silhouette_score(self.feature, cluster_labels)
        print(
            f"The silhouette score for {num_clusters} clusters is: {silhouette_avg}"
        )

        # Calculate the silhouette score for featured data
        sample_silhouette_values = silhouette_samples(self.feature, cluster_labels)

        y_lower = 10
        for i in range(num_clusters):
            # Gathering silhouette scores
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.viridis(float(i)/num_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Cluster labels on y-axis
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute new y for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            
        # Plot title/x and y labels
        ax1.set_title(f"Silhouette Plot for {num_clusters} Clusters")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Assignment")

        # Vertical line to indicate average silhouette score across clusters
        ax1.axvline(x=silhouette_avg, color="red", linestyle="dashdot",linewidth=4)
        
        # Remove the y-axis labels and tick marks
        ax1.set_yticks([])

        if show_clustering:
            fig, ax2 = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)
            
            # creating a color map for clustering points
            color = cm.viridis(cluster_labels.astype(float)/num_clusters)
            ax2.scatter(
                self.feature[:, 0], self.feature[:, 1], marker=".", s=60, lw=0, alpha=0.7, c=color,edgecolor="k"
            )

            # Gathering cluster centers using scikit learns built in function
            centers = clusterer.cluster_centers_
            # Cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="H",
                c="red",
                alpha=1,
                s=600,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
            

            ax2.set_title(f"Visualization of Clustering Assignment for {num_clusters} clusters")
            ax2.set_xlabel("Feature Space 1")
            ax2.set_ylabel("Feature Space 2")
