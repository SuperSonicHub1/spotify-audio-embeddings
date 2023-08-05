"""
TODO: Find a better way of storing embeddings.
sqltie-vss is far too immature for a dataset of my size: https://github.com/asg017/sqlite-vss#disadvantages
Chroma might do it, but it's docs are a tad confusing: https://github.com/chroma-core/chroma
Oh come on: https://docs.trychroma.com/telemetry
"""

import openl3
import soundfile as sf
import numpy as np
import skimage.measure
import tensorflow as tf
from itertools import chain
from more_itertools import ichunked
from files import previews_dir, db_path
from io import BytesIO
import sqlite3
from contextlib import closing
import apsw

conn = apsw.Connection(str(db_path))

# The authors use max-pooling in their paper, so we'll do the same to create a single embedding
# http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf
def max_pool(embeddings):
	return skimage.measure.block_reduce(embeddings, (EMBEDDING_SIZE, 1), np.max)[0]

EMBEDDING_SIZE = 512
model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=EMBEDDING_SIZE)

audio_paths = list(previews_dir.glob("*.mp3"))
sound_files = (sf.read(audio_file) for audio_file in audio_paths)

BATCH_SIZE = 16
for i, sound_files in enumerate(ichunked(sound_files, BATCH_SIZE)):
	sound_files = list(sound_files)
	audios = [sound_file[0] for sound_file in sound_files]
	sample_rates = [sound_file[1] for sound_file in sound_files]

	emb_list, _ = openl3.get_audio_embedding(audios, sample_rates, batch_size=BATCH_SIZE, model=model, frontend='kapre')
	max_pooled = (max_pool(embeddings) for embeddings in emb_list)
	with closing(conn.cursor()) as cursor:
		with conn:
			for (file, embedding) in zip(audio_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], max_pooled):
				outfile = BytesIO()
				np.save(outfile, embedding.astype(np.double), allow_pickle=False)
				cursor.execute("INSERT INTO songs_vectors VALUES (?, ?);", (file.stem, outfile.getvalue()))
		
			
