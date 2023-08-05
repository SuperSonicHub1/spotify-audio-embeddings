from pathlib import Path

BASE = Path("/media/kwilliams") / "Drive the Second" / "spotify-audio-embeddings"

previews_dir = BASE / "previews"
previews_dir.mkdir(exist_ok=True)

embeddings_dir = BASE / "embeddings"
embeddings_dir.mkdir(exist_ok=True)

metadata_dir = BASE / 'metadata'
metadata_dir.mkdir(exist_ok=True)
db_path = metadata_dir / "songs.db"
genres_path = metadata_dir / "spotify_genres.json"
kmeans_song_load_order_path = metadata_dir / 'kmeans_song_load_order.json'

clustering_dir = BASE / 'clustering'
clustering_dir.mkdir(exist_ok=True)
kmeans_path = clustering_dir / 'kmeans.skops'
reduced_embeddings_path = clustering_dir / 'reduced_embeddings.npy'
