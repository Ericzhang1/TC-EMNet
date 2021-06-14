from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from sklearn.manifold import TSNE

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    #print(y_true.shape, y_pred.shape)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)

def k_means(args, hidden, y_true):
    cluster = 3 if args.data_type == 0 else 6
    kmeans = KMeans(init="k-means++", n_clusters=cluster, n_init=10,
                random_state=0).fit(hidden)
    #formatted_dataset = to_time_series_dataset(hidden)
    #kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw").fit(formatted_dataset)

    np.save(f'{args.output_dir}/labels_{args.fold}_{args.data_type}_{args.seed}.npy', kmeans.labels_)
    return purity_score(y_true, kmeans.labels_), adjusted_mutual_info_score(y_true, kmeans.labels_), adjusted_rand_score(y_true, kmeans.labels_)

if __name__ == '__main__':
    pass