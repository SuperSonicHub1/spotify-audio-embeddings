import json
import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
from sklearn.cluster import KMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
from sklearn.decomposition import PCA
import skops.io as sio
from files import (
	genres_path,
	kmeans_song_load_order_path,
	kmeans_path,
	reduced_embeddings_path,
	db_path
)
import apsw

conn = apsw.Connection(str(db_path))
song_vectors_ids = list(conn.execute("SELECT rowid, song_id FROM songs_vectors;"))
rowids = [row[0] for row in song_vectors_ids]
song_ids = [row[1] for row in  song_vectors_ids]

vectors = np.asarray(
	[
		np.load(conn.blobopen('main', 'songs_vectors', 'vector', rowid, False))
		for rowid
		in rowids
	],
	dtype=np.double
)

with genres_path.open() as f:
	genres = json.load(f)

reduced_embeddings = PCA(n_components=2).fit_transform(vectors)
kmeans = KMeans(init="k-means++", n_clusters=len(genres), n_init=4).fit(reduced_embeddings)

with open(kmeans_song_load_order_path, 'w') as f:
	json.dump(song_ids, f)

sio.dump(kmeans, kmeans_path)
np.save(reduced_embeddings_path, reduced_embeddings)
