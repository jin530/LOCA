# LOCA
This is the official code with preprocessed datasets for the WSDM 2021 paper: [`Local Collaborative Autoencoders`.](https://arxiv.org/abs/2103.16103)

The slides can be found [here](https://www.slideshare.net/ssuser1f2162/local-collaborative-autoencoders-wsdm2021).

---

## Dataset

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-dvpl"># Users</th>
    <th class="tg-dvpl"># Items</th>
    <th class="tg-dvpl"># Ratings</th>
    <th class="tg-dvpl">Sparsity</th>
    <th class="tg-dvpl">Concentration</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">ML10M</td>
    <td class="tg-dvpl">69,878</td>
    <td class="tg-dvpl">10,677</td>
    <td class="tg-dvpl">10,000,054</td>
    <td class="tg-dvpl">98.66%</td>
    <td class="tg-dvpl">48.04%</td>
  </tr>
  <tr>
    <td class="tg-0pky">ML20M</td>
    <td class="tg-dvpl">138,493</td>
    <td class="tg-dvpl">26,744</td>
    <td class="tg-dvpl">20,000,263</td>
    <td class="tg-dvpl">99.46%</td>
    <td class="tg-dvpl">66.43%</td>
  </tr>
  <tr>
    <td class="tg-0pky">AMusic</td>
    <td class="tg-dvpl">4,964</td>
    <td class="tg-dvpl">11,797</td>
    <td class="tg-dvpl">97,439</td>
    <td class="tg-dvpl">99.83%</td>
    <td class="tg-dvpl">14.93%</td>
  </tr>
  <tr>
    <td class="tg-0pky">AGames</td>
    <td class="tg-dvpl">13,063</td>
    <td class="tg-dvpl">17,408</td>
    <td class="tg-dvpl">236,415</td>
    <td class="tg-dvpl">99.90%</td>
    <td class="tg-dvpl">16.40%</td>
  </tr>
  <tr>
    <td class="tg-0pky">Yelp</td>
    <td class="tg-dvpl">25,677</td>
    <td class="tg-dvpl">25,815</td>
    <td class="tg-dvpl">731,671</td>
    <td class="tg-dvpl">99.89%</td>
    <td class="tg-dvpl">22.78%</td>
  </tr>
</tbody>
</table>
<br>
We use five public benchmark datasets: MovieLens 10M (ML10M), MovieLens 20M (ML20M), Amazon Digital Music (AMusic), Amazon Video Games (AGames), and Yelp 2015 (Yelp) datasets. We convert all explicit ratings to binary values, whether the ratings are observed or missing. For the MovieLens datasets, we did not modify the original data except for binarization. For the Amazon datasets, We removed users with ratings less than
10, resulting in 97,439 (Music) and 236,415 (Games) ratings. For the Yelp dataset, we pre-processed Yelp 2015 challenge dataset as in <A href='https://github.com/hexiangnan/sigir16-eals'> Fast Matrix Factorization for Online Recommendation with Implicit Feedback </A>, where users and items with less than 10 interactions are
removed.
<br>
<br>

You can get the original datasets from the following links:
<!-- Movielens -->
Movielens: https://grouplens.org/datasets/movielens/

<!-- Amazon review -->
Amazon Review Data: https://nijianmo.github.io/amazon/

<!-- Yelp -->
Yelp 2015: https://github.com/hexiangnan/sigir16-eals/tree/master/data

---

## Basic Usage
- Change the experimental settings in `main_config.cfg` and the model hyperparameters in `model_config`. </br>
- Run `main.py` to train and test models. </br>
- Command line arguments are also acceptable with the same naming in configuration files. (Both main/model config)

For example: ```python main.py --model_name MultVAE --lr 0.001```

## Running LOCA
Before running LOCA, you need (1) user embeddings to find local communities and (2) the global model to cover users who are not considered by local models. </br>

1. Run single MultVAE and EASE to get user embedding vectors and the global model: 

`python main.py --model_name MultVAE` and `python main.py --model_name EASE`

2. Train LOCA with the specific backbone model:

`python main.py --model_name LOCA_VAE` and `python main.py --model_name LOCA_EASE` 

---

## Requirements
- Python 3
- Torch 1.5

## Citation
Please cite our papaer:
```
@inproceedings{DBLP:conf/wsdm/ChoiJLL21,
  author    = {Minjin Choi and
               Yoonki Jeong and
               Joonseok Lee and
               Jongwuk Lee},
  title     = {Local Collaborative Autoencoders},
  booktitle = {{WSDM} '21, The Fourteenth {ACM} International Conference on Web Search
               and Data Mining, Virtual Event, Israel, March 8-12, 2021},
  pages     = {734--742},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3437963.3441808},
  doi       = {10.1145/3437963.3441808},
  timestamp = {Wed, 07 Apr 2021 16:17:44 +0200},
  biburl    = {https://dblp.org/rec/conf/wsdm/ChoiJLL21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
