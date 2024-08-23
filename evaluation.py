#!/usr/bin/env python3

import argparse, json
from utils.emo2vec import emo2vec_sim
from utils.arousal_valance import arousal_valence_sim


def main(input_json_path, output_json_path, uttwise_score=False):

    with open(input_json_path, "r") as f:
        data = json.load(f)

    audio_pairs = [(pair["ref_audio"], pair["gen_audio"]) for pair in data["pairs"]]
    ref_paths = [pair["ref_audio"] for pair in data["pairs"]]
    gen_paths = [pair["gen_audio"] for pair in data["pairs"]]
    arousal_valance_results = arousal_valence_sim(ref_paths, gen_paths)
    emo2vec_results = emo2vec_sim(ref_paths, gen_paths)

    results = []
    scores = [0] * 4

    for arousal_valance_result, emo2vec_result in zip(
        arousal_valance_results, emo2vec_results
    ):
        assert arousal_valance_result[0] == emo2vec_result[0]
        assert arousal_valance_result[1] == emo2vec_result[1]

        ref_path = arousal_valance_result[0]
        gen_path = arousal_valance_result[1]

        emo2vec_sim_utt = emo2vec_result[2]
        emo2vec_sim_frame = emo2vec_result[3]
        arousal_valance_sim_utt = arousal_valance_result[2]
        arousal_valance_sim_frame = arousal_valance_result[3]

        scores[0] += emo2vec_sim_utt
        scores[1] += emo2vec_sim_frame
        scores[2] += arousal_valance_sim_utt
        scores[3] += arousal_valance_sim_frame

        if uttwise_score:
            results.append(
                {
                    "ref_audio": ref_path,
                    "gen_audio": gen_path,
                    "emo2vec_sim_utt": float(emo2vec_sim_utt),
                    "arousal_valance_sim_utt": float(arousal_valance_sim_utt),
                }
            )
        else:
            results.append(
                {
                    "ref_audio": ref_path,
                    "gen_audio": gen_path,
                    "emo2vec_sim_frame": float(emo2vec_sim_frame),
                    "arousal_valance_sim_frame": float(arousal_valance_sim_frame),
                }
            )

    num_pairs = len(data["pairs"])
    scores = [score / num_pairs for score in scores]

    if uttwise_score:
        average_scores = {
            "emo2vec_sim_utt": f"{float(scores[0]):.3f}",
            "arousal_valance_sim_utt": f"{float(scores[2]):.3f}",
        }
    else:
        average_scores = {
            "emo2vec_sim_frame": f"{float(scores[1]):.3f}",
            "arousal_valance_sim_frame": f"{float(scores[3]):.3f}",
        }

    out_json = {"average_scores": average_scores, "results": results}

    with open(output_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate emotion metrics between reference and generated audio."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help="Json file containing the reference and generated audio paths.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Output json path to log the results.",
        default="results.json",
    )
    parser.add_argument(
        "--uttwise_score",
        action="store_true",
        help="whether to output the utterance-wise score instead of frame-wise.",
    )

    args = parser.parse_args()

    main(args.input_json, args.output_json, args.uttwise_score)
