"""Collect results/*.json into one table: mean ± std over seeds per model.

Example:
    python src/summarize.py --out_dir results
"""
import argparse
import json
from glob import glob

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results')
    args = p.parse_args()

    rows = []
    for path in glob(f'{args.out_dir}/*.json'):
        with open(path) as f:
            r = json.load(f)
        rows.append({k: r[k] for k in ['model', 'seed', 'n_params', 'best_epoch', 'val_acc', 'test_acc']})
    df = pd.DataFrame(rows)

    table = df.groupby('model').agg(
        n_params=('n_params', 'first'),
        seeds=('seed', 'count'),
        val_mean=('val_acc', 'mean'), val_std=('val_acc', 'std'),
        test_mean=('test_acc', 'mean'), test_std=('test_acc', 'std'),
    )
    print(table.round(3).to_string())


if __name__ == '__main__':
    main()
