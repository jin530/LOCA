# LOCA
This is the code for the WSDM 2021 paper: `Local Collaborative Filtering`. 

---

## Dataset
You can get preprocessed datasets in here.
<!-- Preprocessed Datasets -->
https://drive.google.com/drive/folders/1DqchJ1RR2TZRNoVeU3MXcXLcMJG0fia_?usp=sharing

You can get the original datasets in following links:
<!-- Movielens -->
Movielens: https://grouplens.org/datasets/movielens/

<!-- Amazon review -->
Amazon Review Data: https://nijianmo.github.io/amazon/

<!-- Yelp -->
Yelp 2016: https://github.com/hexiangnan/sigir16-eals/tree/master/data

---

## Usage
You can train and test the models by running `main.py`.
You can change the datasets and other settings in `main_config.cfg` or in command line.
You can change the model settings in `./model_config/AAA.cfg` or in command line. (AAA is a model name.)

For example: `python main.py --model_name MultVAE --lr 0.001`

Before run the LOCA, you need 1) user embeddings to find local communites and 2) global model to cover the users who are uncovered by local models. 
There are two steps to get the user embeddings.
1. Run single MultVAE and EASE: 

`python main.py --model_name MultVAE` and `python main.py --model_name EASE`

After training the model, trained model will be saved to `./saves/MultVAE/X_YYYYYYYY-ZZZZ` and `./saves/EASE/X_YYYYYYYY-ZZZZ`(X, Y, Z are run_number, date and time, respectively.)

2. Extract embeddings and global model: 

`python main_extract.py --path ./saves/MultVAE/X_YYYYYYYY-ZZZZ` and `python main_extract.py --path ./saves/EASE/X_YYYYYYYY-ZZZZ`

3. Train the LOCA: 

`python main.py --model_name LOCA_VAE` and `python main.py --model_name LOCA_EASE` 

---

## Requirements
- Python 3
- Torch 1.5

<!-- ## Citation
Please cite our papaer:
```
@inproceedings{
} 
``` -->