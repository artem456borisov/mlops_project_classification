# AI DETECTOR

The point of this project is to determine whether a given text is human or machine generated.

## Input and output Data Format

1. input file - .pkl file of pandas DataFrame that must contain:
    - column 'embeddings': torch.tensor(embedding_length, dtype=torch.float32) -> input.shape = torch.Size([embedding_length])
    - column 'label': int, 0 or 1 (0 - human_text, 1 - machine_text). If used for prediction, then just fill with 0.

2. output file (predictions only): .txt
    - every line a prediction (human_text/machine_text)

I used bge-m3 for getting text embeddings with truncation=500. Any fix-length embeddings can be used, but be sure to set the input dim of model (config/model/basic.nn/input_dim)

## Metrics

Classic classification metrics: precision, recall and f1

## Validation

The split was performed based on cohere_0 column of M4GT dataset. When running pipelines, be sure to split train/valid/test/predict into seperate .pkl files. Their location can be specified in config/data

## Dataset

[M4GT](https://github.com/mbzuai-nlp/M4GT-Bench). I only used dataset from subtaskA.

## Modeling

Baseline: basic settings, sgd optimizer.
Main Model: AdamW optimizer, bigger hidden_dim.

## Deployment

Training: 
```poetry run python ./ai_detector/trainer.py train```

Inference: 
```poetry run python ./ai_detector/trainer.py infer```