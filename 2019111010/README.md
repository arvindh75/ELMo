# Advanced NLP Assignment - 2 (ELMo)

#### Arvindh A, 2019111010

## Report
All the plots and figures are compiled in `./Report.pdf`

## Parsing
- Download the dataset from [here](https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz) and move it to `./data` and then unzip it
- The given data is parsed for our purpose using `./parse_data.ipynb` which generates few text files and `corpus.json` which will serve as the corpus for the model training.
- Pre-generated `corpus.json` is already present under `./data`


## Training and Testing
- Download all the files (`tokenized.json`, `model.pt`) from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/arvindh_a_research_iiit_ac_in/EhWpGI08MzNMqWIj0xrbMKABTKNRoXvTcJzSP4bZbW7BaA?e=hkev4g) into `./checkpoints` directory.
- Both the training and testing of the model is handled by `./elmo.ipynb`. `MODE` variable ("TRAIN", "TEST") at the beginning of the notebook determines what mode the notebook will be run on. If the notebook is run in "TEST" mode, you need to manually skip the Training section (indicated by a markdown heading) but run the rest sequentially.
- The sentences and the words associated with them which you require the embedding for can be changed as well. At last, it plots a heatmap to visualize the similarities between word embeddings.

## Remarks
The training process is quite heavy computationally. It took around 35 hrs for 50 epochs on a 2080Ti.
