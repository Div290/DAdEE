# DAdEE: Unsupervised Domain Adaptation in Early Exit PLMs
This repository is the official implementation of the work DAdEE: Unsupervised Domain Adaptation in Early Exit PLMs

## Requirements

We built upon our code using the [huggingface transformers](https://huggingface.co/docs/transformers/en/index).

## Training 

To fine-tune a pre-trained language model on and train the internal classifiers follow the command:

```Training, adaptation and Inference
python3 main.py --pretrain --adapt --src books --tgt dvd
```

## Code acknowledgement
We acknowledge the [bert-aad](https://github.com/bzantium/bert-AAD/blob/master/README.md) repository and thank them for making the source code publically available 
