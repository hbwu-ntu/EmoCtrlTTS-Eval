# EmoCtrl-TTS evaluation tool
This repository maintains codes used for emotion-related metrics for the paper "Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching-Based Zero-Shot Text-to-Speech". 

[Paper](https://arxiv.org/abs/2407.12229)  [Demo](https://www.microsoft.com/en-us/research/project/emoctrl-tts/)
## Overview
We present two types of emotion-related metrics based on two emotion embedding extractors, [emotion2vec](https://github.com/ddlBoJack/emotion2vec) and [arousal-valence feature extractor](https://github.com/audeering/w2v2-how-to). 
- (Default) Frame (chunk)-wise emotion similarity of two utterances. The score is the average of frame (chunk)-wise computation.

For details, please refer to Section 4.2 of [EmoCtrl-TTS](https://arxiv.org/abs/2407.12229).
These proposed emotion-related objective metrics can serve as benchmarks for future research aiming to assess the expressiveness of generated emotional speech.

## Prerequisites
- Linux
  - python 3.8

## Installation
```sh
pip install -r requirements.txt
pip install -U funasr
```

## How to use the evaluation metrcis
Prepare the `input.json` containing the generated and reference utterance paths. The current prepared samples are generated utterances.
```sh
$ python evaluation.py \
    --input_json ./samples/input.json \
    --output_json ./samples/output.json
```
The results (for `average_scores` and each evaluation pair) will be shown in `./samples/output.json`. 

## Reference
- https://github.com/ddlBoJack/emotion2vec
- https://github.com/audeering/w2v2-how-to

## License
[MIT Licence](LICENSE.txt)

## Citation
Please cite our paper if you find our repository is useful.
```
@article{wu2024laugh,
  title={Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching-Based Zero-Shot Text-to-Speech},
  author={Wu, Haibin and Wang, Xiaofei and Eskimez, Sefik Emre and Thakker, Manthan and Tompkins, Daniel and Tsai, Chung-Hsien and Li, Canrun and Xiao, Zhen and Zhao, Sheng and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2407.12229},
  year={2024}
}

@article{kanda2024making,
  title={Making Flow-Matching-Based Zero-Shot Text-to-Speech Laugh as You Like},
  author={Kanda, Naoyuki and Wang, Xiaofei and Eskimez, Sefik Emre and Thakker, Manthan and Yang, Hemin and Zhu, Zirun and Tang, Min and Li, Canrun and Tsai, Steven and Xiao, Zhen and others},
  journal={arXiv preprint arXiv:2402.07383},
  year={2024}
}
```