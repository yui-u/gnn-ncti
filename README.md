# Non-chord Tone Identification Using GNN for Improving Unsupervised Harmonic Analysis

This repository is the implementation of 
"Non-chord Tone Identification Using GNN for Improving Unsupervised Harmonic Analysis." (Uehara and Tojo, Accepted to APSIPA ASC 2025)

## Requirements
- Python 3.10.12
- Required packages are listed in: [requirements.txt](requirements.txt)

## Trained models
- [trained_models](trained_models)

## Datasets
### J. S. Bach's Little Organ Book
Download files from [https://github.com/yui-u/little-organ-analyses](https://github.com/yui-u/little-organ-analyses).
Place all `.musicxml` and `.rntxt` files in a directory as below.
```commandline
little-organ-book-rntxt
├── BWV599_v2_analysis.rntxt
├── BWV599_v2_score.musicxml
├── BWV600_v2_analysis.rntxt
├── BWV600_v2_score.musicxml
...
```

### Chorales
Download files named `analysis.txt` from [https://github.com/MarkGotham/When-in-Rome/tree/master/Corpus/Early_Choral/Bach%2C_Johann_Sebastian/Chorales](https://github.com/MarkGotham/When-in-Rome/tree/master/Corpus/Early_Choral/Bach%2C_Johann_Sebastian/Chorales).
Rename `analysis.txt` to `ChoralesXXX_analysis.rntxt` and place all files in a directory as below.
```commandline
Chorales
├── Chorales001_analysis.rntxt
├── Chorales002_analysis.rntxt
├── Chorales003_analysis.rntxt
...
```

### The Well-Tempered Clavier
Download files named `analysis.txt` from [https://github.com/MarkGotham/When-in-Rome/tree/master/Corpus/Keyboard_Other/Bach%2C_Johann_Sebastian/The_Well-Tempered_Clavier_I](https://github.com/MarkGotham/When-in-Rome/tree/master/Corpus/Keyboard_Other/Bach%2C_Johann_Sebastian/The_Well-Tempered_Clavier_I).
Rename `analysis.txt` to `The_Well-Tempered_Clavier_IXX_analysis.rntxt` and `score.mxl` to `The_Well-Tempered_Clavier_IXX_score.mxl`; place all files in a directory as below.
```commandline
wir_flatten/WTC_I
├── The_Well-Tempered_Clavier_I01_analysis.rntxt
├── The_Well-Tempered_Clavier_I01_score.mxl
├── The_Well-Tempered_Clavier_I02_analysis.rntxt
├── The_Well-Tempered_Clavier_I02_score.mxl
...
```

## Generate a preprocessed dataset
### full
```
python run.py \
preprocess_dataset \
--dir_output dataset \
--dataset <Path to the directory containing rntxt (and mxl or musicxml)> \
--cv_num_set 5 \
--chord_type full
```

### triad+dominant
```
python run.py \
preprocess_dataset \
--dir_output dataset \
--dataset <Path to the directory containing rntxt (and mxl or musicxml)> \
--cv_num_set 5 \
--chord_type triad+dominant
```

## Example of inference with a trained model
```
python run.py \
inference_nct \
--dir_output inference \
--dir_preprocessed_dataset dataset/little-organ-book-rntxt-cvn5-halfbeat-triad+dominant-nctrth0.5 \
--cv_set_no 4 \
--dir_model trained_models/out-little-organ/gatv2-mp4-head3-normalized/nct-checkpoint-cv4-seed123
```

## Train an NCT identification model from scratch

### Train an NCT identification model (GraphSAGE, 3-layer, cross-validation-setno=0)
```
python run.py \
train_nct \
--dir_output out-little-organ/sage-mp4-seed123 \
--num_epochs 1024 \
--preprocessed_dataset_path dataset/little-organ-book-rntxt-cvn5-halfbeat-triad+dominant-nctrth0.5 \
--device cuda:0 \
--seed 123 \
--cv_set_no 0 \
--gnn_model_type sage \
--gnn_activation_fn relu \
--num_message_passing 3 \
--metric fscore \
--fscore_beta 1.0 \
--gradient_clip 1.0 \
--learning_rate 1e-3 \
--patience -1
```

### Train an NCT identification model (GATv2, 4-layer, 3-heads, cross-validation-setno=0)
```
python run.py \
train_nct \
--dir_output out-little-organ/gatv2-mp4-head3-seed123 \
--num_epochs 1024 \
--preprocessed_dataset_path dataset/little-organ-book-rntxt-cvn5-halfbeat-triad+dominant-nctrth0.5 \
--device cuda:0 \
--seed 123 \
--cv_set_no 0 \
--gnn_model_type gatv2 \
--gnn_activation_fn relu \
--num_message_passing 4 \
--num_gat_heads 3 \
--metric fscore \
--fscore_beta 1.0 \
--gradient_clip 1.0 \
--learning_rate 1e-3 \
--patience -1
```

## Acknowledgments
This work was in part supported by grants from JSPS KAKENHI Grant Numbers 23K20011, 25H01169, and JST ACT-X Grant Number JPMJAX24C6.
We used ABCI 3.0 provided by AIST and AIST Solutions with support from ''ABCI 3.0 Development Acceleration Use''.

## Publications
Please cite the following paper when using this code:
```
Y. Uehara and S. Tojo, 
"Note-level Nonchord-tone Identification with Graph Neural Networks", 
2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (Accepted).
```