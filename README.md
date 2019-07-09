# Adaptive Attention Span for Transformers

This is a code for running experiments in [Adaptive Attention Span for Transformers](https://arxiv.org/abs/1905.07799) paper. It trains a Transformer model on character-level language modeling tasks. The adaptive span allows a model to learn an optimal context size for each self-attention head from training data. As shown in the below figure, only few heads require long attention span, thus making it possible to increase the context size to 8k tokens without increasing computation time and memory footprint significantly.

<div align="center">
  <img src="README_files/span.png" width="400px" />
</div>

## Requirements
You need PyTorch 0.4.1 or above and a cuda-enabled GPU to run the code.

## Running experiments in the paper
Scripts for running experiments in the paper are located in `./experiments/` directory. For example, a smaller 8-layer version of our model can be trained on a single GPU by running:
```bash
bash experiments/enwiki8_small.sh
```
It should reach about 1.3bpc on dev after 150k steps.

For training larger models, multiple GPUs are recommended. In the script files, you can configure the number of available GPUs. Increase the `--batch-split` argument if you run out of GPU memory (it splits batches into smaller pieces without changing the final result).

We  obtained the following results in our experiments:

| Experiment | #params | dev (bpc) | test (bpc) |
| ---------- | ---:| ---:| ----:|
| enwik8 | 38M | 1.04 | 1.02 |
| enwik8_large | 209M | 1.00 | 0.98 |
| text8 | 39M | 1.05 | 1.11 |
| text8_large | 209M | 1.01 | 1.07 |

## More about the code
- **Multi GPUs and nodes:** By default, the code uses `nn.DataParallel` to utilize all available GPUs. For more efficiency, enable distributed training by `--distributed` argument, which can run on multiple nodes.
- **Base model:** As a base model, the code implements a Transformer model with relative position embeddings and hidden state caching for processing a sequence of tokens.
- **Adaptive attention span:** An argument `--adapt-span` enables adaptive span. Otherwise a model will have a fixed attention span. The adaptive-span is implemented as a `nn.Module` to make it easier to plug it into other models.
- **Training time:** A large model training takes about 1.2sec/batch near the end (initially it's faster because the attention spans are smaller) on 8 V100 GPUs. So, for example, the whole `enwik8_large` training of 170k steps should take less than 2.4 days.

## License
The code is licensed under CC-BY-NC license. See the LICENSE file for more details.

## Acknowledgement
We thank Xavier Martinet for helping with cleaning the code. The data preprocessing scripts are downloaded from [awd-lstm](https://github.com/salesforce/awd-lstm-lm/) and [transformer-XL](https://github.com/kimiyoung/transformer-xl) repos. The `adagrad_with_grad_clip.py` is mostly adapted from PyTorch.
