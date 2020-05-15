# Fashion IQ 2019 Challenge RippleAI Team Solution

This project hosts the code for our paper.

- [Youngjae Yu](https://yj-yu.github.io/home), [Seunghwan Lee](http://rippleai.cc/) [Yunchel Choi](http://rippleai.cc/)  [Gunhee Kim](http://vision.snu.ac.kr/~gunhee/). CurlingNet: Compositional Learning between Images and Text for Fashion IQ Data
. In *ICCV Workshop Linguistics Meets image and video retrieval *, 2019.

[Link to ICCV19 Workshop](https://sites.google.com/view/lingir/fashion-iq)

## Training

Please see the official starter code to download and prepare dataset. [Link to official baseline](https://github.com/XiaoxiaoGuo/fashion-iq/tree/master/start_kit)
Prepare 256x256 resized images in dataset/resized_images , and modify configure files in configs/ce folder.

To reproduce the single best model, we recommend to use W2V [Link](https://www.kaggle.com/jacksoncrow/word2vec-flickr30k/version/1) and set the path to the parameters. Pretraining also helps improve the final performance.

Train the model

```bash
python train.py
```

## Reference

If you use this code or dataset as part of any published research, please refer following paper,

```bibtex
@article{yu2020curlingnet,
  title={CurlingNet: Compositional Learning between Images and Text for Fashion IQ Data},
  author={Yu, Youngjae and Lee, Seunghwan and Choi, Yuncheol and Kim, Gunhee},
  journal={arXiv preprint arXiv:2003.12299},
  year={2020}
}
```

## System Requirements

- Python 3.6
- Pytorch 1.4
- CUDA 10.0 supported GPU with at least 12GB memory

## Acknowledgement

This work was inspired by great prior works for compositing image and text, but in particular coollaborate expert and TIRG.

Collaborative Expert(https://github.com/albanie/collaborative-experts)
TIRG(https://github.com/google/tirg)
