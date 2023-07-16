
# Code for Supporter

> The implementation of our paper accepted by ACL 2023: Facilitating Multi-turn Emotional Support Conversation with Positive Emotion Elicitation: A Reinforcement Learning Approach

<img src="https://img.shields.io/badge/Venue-ACL--23-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/>  <img src="https://img.shields.io/badge/Last%20Updated-2022--12-2D333B" alt="update"/>

## Requirements

- `Python==3.8.0`
- `torch==1.10.0`
- `transformers==4.1.1`
- Download [Blender model 90M](https://huggingface.co/facebook/blenderbot_small-90M), and put it into the `blender` folder

## Dataset

- The preprocessed dataset is already provided at [Google Driven](https://drive.google.com/file/d/1JpR37D_IHt_FYwY49pI-Q4SO7bjXQ6iG/view?usp=sharing). Change the folder name to `data`.

- If you want to create the dataset yourself, download the [comet-atomic-2020 (BART) checkpoint](https://github.com/allenai/comet-atomic-2020) and place it in `/data/ConstructDataset/Comet`. The preprocessing details could be found in the `main.sh` script.

## Rewards

### Emotional Support Reward Model
- Download [Emotion Classification Model](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), and put it into the `emotion` folder

### Dialogue Coherence Reward Models

- The trained Dialogue Coherence Reward Models is already provided at [Google Driven](https://drive.google.com/file/d/1JpR37D_IHt_FYwY49pI-Q4SO7bjXQ6iG/view?usp=sharing).
- Download [bert-base-cased](https://huggingface.co/bert-base-cased), and put it into the `rewards/bert`

- If you want to train the Dialogue Coherence Reward Models yourself:
  ```
  cd rewards
  python construct_dataset.py
  cd ..
  bash main_rewards.sh
  ```

## Training, Testing and Evaluating

```
bash main.py
```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{DBLP:conf/acl/ZhouCWH23,
  author       = {Jinfeng Zhou and
                  Zhuang Chen and
                  Bo Wang and
                  Minlie Huang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Facilitating Multi-turn Emotional Support Conversation with Positive
                  Emotion Elicitation: {A} Reinforcement Learning Approach},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                  July 9-14, 2023},
  pages        = {1714--1729},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.acl-long.96},
  timestamp    = {Thu, 13 Jul 2023 16:47:40 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ZhouCWH23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
