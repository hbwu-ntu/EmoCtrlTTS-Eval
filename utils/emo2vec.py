#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from .utils import audio_emb_sync, similarity


def emo2vec_sim(
    ref_paths,
    gen_paths,
    model_type="iic/emotion2vec_base_finetuned",
    granularity="frame",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=model_type,
        model_revision="v2.0.4",
        device=device,
    )

    with torch.no_grad():
        pred1s = inference_pipeline(ref_paths, granularity=granularity)
        pred2s = inference_pipeline(gen_paths, granularity=granularity)

    results = []

    for res1, res2, ref_path, gen_path in zip(pred1s, pred2s, ref_paths, gen_paths):
        emb1 = torch.tensor(res1["feats"])
        emb2 = torch.tensor(res2["feats"])

        sim_utt = F.cosine_similarity(
            torch.mean(emb1, axis=0).unsqueeze(0), torch.mean(emb2, axis=0).unsqueeze(0)
        ).item()

        emb1 = emb1.cpu().numpy()
        emb2 = emb2.cpu().numpy()
        emb2 = audio_emb_sync(emb2, emb1.shape[0])
        sim_frame = similarity(emb1, emb2)

        results.append([ref_path, gen_path, sim_utt, sim_frame])

    return results
