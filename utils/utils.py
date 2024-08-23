#!/usr/bin/env python3

from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def audio_emb_sync(z, num_audio_frames):
    """
    use interpolation with nearest to sync the phone frames with the audio frames
    args:
        z: the frame-level laughter embedding
        num_audio_frames: the number of audio frames
    return:
        z: the frame-level phone transcript, a list of phone ids(int), synced with the audio frames
    """
    # Create an interpolation function based on the old labels
    N = num_audio_frames
    z = np.array(z)
    f = interp1d(np.linspace(0, 1, len(z)), z, axis=0, kind="nearest")

    # Use this function to generate labels for the new frames
    z_sync = f(np.linspace(0, 1, N))
    return z_sync.astype("float32")


def similarity(matrix1, matrix2):
    cosine_similarities = []
    for i in range(matrix1.shape[0]):
        vector1 = matrix1[i].reshape(1, -1)  # Reshape to 2D array with a single row
        vector2 = matrix2[i].reshape(1, -1)
        cosine_sim = cosine_similarity(vector1, vector2)
        cosine_similarities.append(cosine_sim[0, 0])

    average_cosine_similarity = np.mean(cosine_similarities)
    return average_cosine_similarity
