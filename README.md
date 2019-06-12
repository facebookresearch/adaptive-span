# Adaptive Attention Span for Transformers

## ToDo
- [ ] make the attention span class modular (remove args etc.)
- [ ] improve code style
- [ ] all experiments from the paper
- [ ] write readme
- [ ] remove internal experiment files

### Training of a model
A model can be trained
```bash
python main.py --data /path/to/data --checkpoint /path/to/checkpoint
```
Add `--plot` argument to plot training curves in Visdom.
<<<<<<< HEAD

A model with adaptive span can be trained with:
```
python main.py --data /path/to/data --checkpoint /path/to/checkpoint --attn-span-loss 0.000002
```

=======

A model with adaptive span can be trained with:
```
python main.py --data /path/to/data --checkpoint /path/to/checkpoint --attn-span-loss 0.000002
```

>>>>>>> master
### Evaluation of a model
After training, a model can be tested on the whole test data by adding `--full-test`:
```bash
python main.py [same arguments as training] --full-test --batch-sz 1
```
This will print the validation and test performances of the model.
A larger batch size can be used, but it will slightly worsen the performance because more samples will lack valid cache.

## Running experiments in the paper
Scripts for running the experiments in the paper can be found in `experiments` folder.
### In internal Slurm (remove later)
The following will launch an experiment in Slurm that should reach 1.01bpc on validation.
```bash
bash experiments/slurm_enwiki8_large.sh
```