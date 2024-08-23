#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import librosa
from .utils import audio_emb_sync, similarity


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config=None):

        super().__init__()

        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(1024, 3)

    def forward(self, features, **kwargs):

        x = torch.mean(features, dim=1)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        hidden = self.dropout(x)
        x = self.out_proj(hidden)

        return x, hidden


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        all_hidden_states = outputs[0]
        hidden_states = torch.mean(all_hidden_states, dim=1)
        logits, hidden = self.classifier(all_hidden_states)

        return hidden_states, logits, all_hidden_states


class EmoModelWrapper:
    def __init__(
        self,
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cuda",
    ):

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    def forward(self, audio, sampling_rate=16000, chunk_window=25, chunk_hop=12):
        """
        args:
            audio: np.array, audio data;
                egs: np.zeros((sampling_rate), dtype=np.float32)
            sampling_rate: int, sample rate of audio
            chunk_window: int, window size of chunk, -1 means no chunk
            chunk_hop: int, hop size of chunk
        """

        self.model.eval()

        y = self.processor(audio, sampling_rate=sampling_rate)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.model.device)

        with torch.no_grad():
            hidden_states, logits, all_hidden_states = self.model(y)

            with torch.no_grad():
                emo_logits_utt = self.model.classifier(
                    all_hidden_states
                )  # shape [1, 3]

            chunks = []
            num_chunks = (all_hidden_states.size(1) - chunk_window) // chunk_hop + 1

            for i in range(num_chunks):
                start = i * chunk_hop
                end = start + chunk_window
                chunk = all_hidden_states[:, start:end, :]
                chunks.append(chunk)

            chunked_wav2vec2_emotion = torch.stack(chunks, dim=1).transpose(2, 1)
            with torch.no_grad():
                emo_logits, emo_hidden = self.model.classifier(
                    chunked_wav2vec2_emotion
                )  # shape [1, num_chunks, 3]
                emo_logits_chunk = emo_logits.squeeze(0).cpu().numpy()
                emo_hidden = emo_hidden.squeeze(0).cpu().numpy()

        return emo_logits_utt[0], emo_logits_chunk, hidden_states


def arousal_valence_sim(ref_paths, gen_paths):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emo_model = EmoModelWrapper(device=device)

    results = []

    for ref_path, gen_path in zip(ref_paths, gen_paths):
        ref_audio, _ = librosa.load(ref_path, sr=16000)
        gen_audio, _ = librosa.load(gen_path, sr=16000)

        ref_emo_logits_utt, ref_emo_logits_chunk, _ = emo_model.forward(ref_audio)
        gen_emo_logits_utt, gen_emo_logits_chunk, _ = emo_model.forward(gen_audio)

        ref_emo_logits_utt = ref_emo_logits_utt.cpu().numpy()[:, [0, 2]] - 0.5
        gen_emo_logits_utt = gen_emo_logits_utt.cpu().numpy()[:, [0, 2]] - 0.5

        gen_emo_logits_chunk = audio_emb_sync(
            gen_emo_logits_chunk, ref_emo_logits_chunk.shape[0]
        )
        ref_emo_logits_chunk = ref_emo_logits_chunk[:, [0, 2]] - 0.5
        gen_emo_logits_chunk = gen_emo_logits_chunk[:, [0, 2]] - 0.5

        utt_sim = similarity(ref_emo_logits_utt, gen_emo_logits_utt)
        chunk_sim = similarity(ref_emo_logits_chunk, gen_emo_logits_chunk)

        results.append([ref_path, gen_path, utt_sim, chunk_sim])

    return results
