"""
Based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
"""

from sklearn.cluster import KMeans
import skops.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import mplcursors
import json
import pygame
from files import (
	kmeans_song_load_order_path,
	db_path,
	kmeans_path,
	reduced_embeddings_path,
	previews_dir,
)
import apsw

with kmeans_song_load_order_path.open() as f:
	song_ids = json.load(f)

conn = apsw.Connection(str(db_path))

songs = [
	conn.execute(
		"""
WITH
	artists as (
		SELECT
			group_concat(artists.value ->> '$.name', ' | ') as artist
		FROM
			songs,
			json_each(songs.info, '$.artists') as artists
		WHERE
			songs.id = 'spotify:track:6Sq7ltF9Qa7SNFBsV5Cogx'
	)
SELECT
	songs.id,
	songs.info ->> '$.name' as title,
	artists.artist,
	group_concat(songs_genres.genre, ', ') as genres
FROM
	songs,
	artists
LEFT JOIN
	songs_genres ON songs.id = songs_genres.song_id
WHERE
	songs.id = 'spotify:track:6Sq7ltF9Qa7SNFBsV5Cogx';
""",
		{'song_id': song_id}
	).fetchone()
	for song_id
	in song_ids
]

kmeans: KMeans = sio.load(kmeans_path)
reduced_embeddings = np.load(reduced_embeddings_path)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure(1)
plt.clf()
plt.imshow(
	Z,
	interpolation="nearest",
	extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	cmap=plt.cm.Paired,
	aspect="auto",
	origin="lower",
)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
	centroids[:, 0],
	centroids[:, 1],
	marker="x",
	s=169,
	linewidths=3,
	color="w",
	zorder=10,
)
points = plt.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], "k.", markersize=2, zorder=20)
cursor = mplcursors.cursor(points, hover=True)

@cursor.connect("add")
def on_add(sel):
	index = sel.index
	region = kmeans.labels_[index]
	song = songs[index]

	text = f"""{song[1]!r} by {song[2]} ({song[0]})
Region {region}
"""
	sel.annotation.set_text(text)

	pygame.mixer.music.load(previews_dir / f"{song[0]}.mp3")
	pygame.mixer.music.play(-1)

def on_key_press(event):
	if event.key == ' ':
		if pygame.mixer.music.get_busy():
			pygame.mixer.music.pause()
		else:
			pygame.mixer.music.unpause()
	elif event.key == 'm':
		if pygame.mixer.music.get_volume() != 0:
			pygame.mixer.music.set_volume(0)
		else:
			pygame.mixer.music.set_volume(1)

fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.title(
	"K-means clustering on the Spotify embeddings dataset (PCA-reduced data)\n"
	"Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
pygame.init()
pygame.mixer.init()
plt.show()


