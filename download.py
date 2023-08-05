
from spotipy import Spotify
from requests import Session
from selectolax.parser import HTMLParser
import json
from datetime import datetime
import apsw
from contextlib import closing
from more_itertools import ichunked
from itertools import chain, islice
from tqdm import tqdm
from urllib.request import urlretrieve
from multiprocessing import Pool
from collections import namedtuple
from files import previews_dir, db_path, genres_path

FIRST_N_SONGS = 10

conn = apsw.Connection(str(db_path))
# Support multiple sources
conn.execute("""CREATE TABLE IF NOT EXISTS songs (
	id TEXT PRIMARY KEY,
	info JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS songs_vectors (
	song_id TEXT PRIMARY KEY REFERENCES song(id),
	vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS songs_genres (
	song_id TEXT NOT NULL REFERENCES song(id),
	genre TEXT NOT NULL
);""")

session = Session()
access_token_expiration = datetime.now()
_spotify_client = None

def get_spotify_client() -> Spotify:
	global access_token_expiration
	global _spotify_client

	if datetime.now() > access_token_expiration:
		res = session.get("https://open.spotify.com/")
		res.raise_for_status()
		tree = HTMLParser(res.text)
		session_data_node = tree.css_first("script#session")
		assert session_data_node

		session_data = json.loads(session_data_node.text())
		access_token_expiration = datetime.fromtimestamp(session_data['accessTokenExpirationTimestampMs'] / 1000)

		_spotify_client = Spotify(
			session_data['accessToken'],
			requests_session=session
		)

	return _spotify_client

def genre_search(genre: str):
	sp = get_spotify_client()

	LIMIT = 50

	query = f"genre:{genre}"
	offset = 0

	while True:
		songs = sp.search(
			q=query,
			market="US",
			type='track',
			limit=LIMIT,
		)

		tracks = songs['tracks']['items']

		if len(tracks):
			# How are `null`s getting in here? I have no clue.
			yield from (dict(**track, genre=genre) for track in tracks if track and track.get('preview_url'))
		else:
			return

		offset += songs['tracks']['total']

genres = get_spotify_client().recommendation_genre_seeds()['genres']
with open(genres_path, 'w') as f:
	json.dump(genres, f)

songs = chain.from_iterable(
	islice(genre_search(genre), FIRST_N_SONGS)
	for genre
	in genres
)

songs = tqdm(songs, total=FIRST_N_SONGS * len(genres))

for song_chunk in ichunked(songs, 10_000):
	song_chunk = list(song_chunk)
	with conn:
		with closing(conn.cursor()) as cursor:
			# Spotify has a suprisingly large number of dupes
			cursor.executemany("INSERT OR IGNORE INTO songs VALUES (?, ?);", (
				(song['uri'], json.dumps(song))
				for song
				in song_chunk
			))
			cursor.executemany("INSERT INTO songs_genres VALUES (?, ?);", (
				(song['uri'], song['genre'])
				for song
				in song_chunk
			))

print("Info download complete!")

def download(song):
	urlretrieve(song["preview_url"], previews_dir / f"{song['id']}.mp3")

with Pool(8) as p:
	songs = conn.execute("SELECT id, info -> '$.preview_url' as preview_url FROM songs;")
	downloads = p.imap_unordered(download, (dict(id=song[0], preview_url=json.loads(song[1])) for song in songs))
	for _ in tqdm(downloads, total=conn.execute("SELECT COUNT(*) AS count FROM songs;").fetchone()[0]):
		pass

print("Preview download complete!")
