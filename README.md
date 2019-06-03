# Adaptive Attention Span for Transformers

## ToDo
- [ ] make the attention span class modular (remove args etc.)
- [ ] improve code style
- [ ] all experiments from the paper
- [ ] write readme
- [ ] remove internal experiment files

## Training

## Testing
```bash
python main.py $args --checkpoint $path/model.pt --load-only --full-test \
--batch-sz $bsz
```

## Running experiments in the paper

### In internal Slurm (remove later)
The following will launch an experiment in Slurm that should reach 1.01bpc on validation.
```bash
bash experiments/slurm_enwiki8_large.sh
```
