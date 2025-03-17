## Data  
Sample train/val/test data is in the `data/` folder. These are the standard datasets from PTB.
First preprocess the data:
```
python preprocess.py --trainfile data/train.txt --valfile data/valid.txt --testfile data/test.txt --outputfile data/ptb --vocabminfreq 1 --lowercase 0 --replace_num 0 --batchsize 16
```
Running this will save the following files in the `data/` folder: `ptb-train.pkl`, `ptb-val.pkl`,
`ptb-test.pkl`, `ptb.dict`. Here `ptb.dict` is the word-idx mapping, and you can change the
output folder/name by changing the argument to `outputfile`. Also, the preprocessing here
will replace singletons with a single `<unk>` rather than with Berkeley parser's mapping rules
(see below for results using this setup).

## Training
To train the URNNG:
```
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path urnng.pt --mode unsupervised --gpu 0
```
where `save_path` is where you want to save the model, and `gpu 0` is for using the first GPU
in the cluster (the mapping from PyTorch GPU index to your cluster's GPU index may vary).
Training should take 2 to 3 days depending on your setup.

To train the RNNG:
```
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path rnng.pt --mode supervised --train_q_epochs 18 --gpu 0 
```

For fine-tuning:
```
python train.py --train_from rnng.pt --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path rnng-urnng.pt --mode unsupervised --lr 0.1 --train_q_epochs 10 --epochs 10 --min_epochs 6 --gpu 0 --kl_warmup 0
```

To train the LM:
```
python train_lm.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --test_file data/ptb-test.pkl --save_path lm.pt 
```

## Evaluation
To evaluate perplexity with importance sampling on the test set:
```
python eval_ppl.py --model_file urnng.pt --test_file data/ptb-test.pkl --samples 1000 --is_temp 2 --gpu 0
```
The argument `samples` is for the number of importance weighted samples, and `is_temp` is for
flattening the inference network's distribution (footnote 14 in the paper).
The same evaluation code will work for RNNG. 

For LM evaluation:
```
python train_lm.py --train_from lm.pt --test_file data/ptb-test.pkl --test 1
```

To evaluate F1, first we need to parse the test set:
```
python parse.py --model_file urnng.pt --data_file data/ptb-test.txt --out_file pred-parse.txt 
--gold_out_file gold-parse.txt --gpu 0
```
This will output the predicted parse trees into `pred-parse.txt`. We also output a version
of the gold parse `gold-parse.txt` to be used as input for `evalb`, since sentences with only trivial spans are ignored by `parse.py`. Note that corpus/sentence F1 results printed here do not correspond to the results reported in the paper, since it does not ignore punctuation. 

Finally, download/install `evalb`, available [here](https://nlp.cs.nyu.edu/evalb).
Then run:
```
evalb -p COLLINS.prm gold-parse.txt test-parse.txt
```
where `COLLINS.prm` is the parameter file (provided in this repo) that tells `evalb` to ignore
punctuation and evaluate on unlabeled F1.
