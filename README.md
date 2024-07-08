# GET

Source code and datasets for the paper "Evidence-aware Fake News Detection with Graph Neural Networks".

## Requirements

We use Pytorch 1.9.1 and python 3.6. Other requirements are in requirements.txt.

```
pip install -r requirements.txt
```
## Log
Create folder **logs/get** to store log.
## Data 

We utilize two widely used datasets. 

Create a new folder named **embedding** to store embedding word.

Go to http://nlp.stanford.edu/data/glove.6B.zip to download embedding word and store to **embedding** folder

## Usage

You can run the commands below to train and test our model on Snopes Dataset.

```
python MasterFC/master_get.py --dataset="Snopes" \
                             --cuda=1 \
                             --fixed_length_left=30 \
                             --fixed_length_right=100 \
                             --log="logs/get" \
                             --loss_type="cross_entropy" \
                             --batch_size=32 \
                             --num_folds=5 \
                             --use_claim_source=0 \
                             --use_article_source=1 \
                             --path="formatted_data/declare/" \
                             --hidden_size=300 \
                             --epochs=100 \
                             --num_att_heads_for_words=5 \
                             --num_att_heads_for_evds=2 \
                             --gnn_window_size=3 \
                             --lr=0.0001 \
                             --gnn_dropout=0.2 \
                             --seed=123756 \
                             --gsl_rate=0.6 \
                             --lamda1=0.4 \
                             --lamda2=0.01 \
                             --use_oc=True
```    

You can also simply run the bash script.
```
sh run.sh  (run all experiments)
```

