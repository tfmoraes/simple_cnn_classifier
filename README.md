# Simple CNN classifier

## Training


```bash
uv run train.py
```

Use `--help` for its parameters:

```bash
> uv run train.py --help
usage: train.py [-h] [-e N] [-b N] [--lr LR]

options:
  -h, --help            show this help message and exit
  -e N, --epochs N      number of total epochs to run (default: 200)
  -b N, --batch-size N  Batch size (default: 128)
  --lr LR, --learning-rate LR
                        Learning rate (default: 0.001)
```
