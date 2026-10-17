"""Dataset statistics of the evaluation data (canonical_v2 copy on HOST_A). Only reads remote files.

Usage (local machine, repository root):  python tables/src/collect_dataset_stats.py
Writes tables/dataset_stats.csv: per dataset the numbers of users, items and interactions, the density, the split sizes
(x_label 0 / 1 / 2 = train / validation / test), and the number of item rows of each feature file whose all-zero vector was
replaced by the mean of the non-zero rows (manifest_v2.csv column n_rows_filled, written by tools/make_data_v2.py), and the feature dimensions (video only for MicroLens),
and the number of users with test targets and of those with exactly one test target (revision log 18-1).
"""
import csv
import os
import subprocess
import sys

DATA = '<RUN_ROOT_B>/P04_data/canonical_v2'
# revision log 18-1 (2026-10-16): all nine datasets of the pool of Section IV-A (before: six, without arts, office, microlens).
DATASETS = ['baby', 'pantry', 'scientific', 'sports', 'clothing', 'instruments', 'arts', 'office', 'microlens']
FIELDS = ['dataset', 'users', 'items', 'interactions', 'density', 'train', 'valid', 'test', 'image_filled', 'text_filled',
          'feature_rows', 'image_dim', 'text_dim', 'video_dim', 'test_users', 'test_users_one']


def npy_shape(path):
    """shape from the .npy header (no numpy on the HOST_A system python)"""
    import ast
    with open(path, 'rb') as fh:
        magic = fh.read(8)
        n = int.from_bytes(fh.read(2 if magic[6] == 1 else 4), 'little')
        return ast.literal_eval(fh.read(n).decode('latin1'))['shape']


def scan():
    fill = {}
    with open(os.path.join(DATA, 'manifest_v2.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            fill[(r['dataset'], r['file'])] = (r['n_rows_filled'], r['n_rows'])
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(FIELDS)
    for ds in DATASETS:
        users, items, split, n, ntest = set(), set(), {'0': 0, '1': 0, '2': 0}, 0, {}
        with open(os.path.join(DATA, ds, f'{ds}.inter')) as fh:
            hdr = fh.readline().rstrip('\n').split('\t')
            iu, ii, il = hdr.index('userID'), hdr.index('itemID'), hdr.index('x_label')
            for line in fh:
                c = line.rstrip('\n').split('\t')
                users.add(c[iu]); items.add(c[ii]); split[c[il]] += 1; n += 1
                if c[il] == '2':
                    ntest[c[iu]] = ntest.get(c[iu], 0) + 1
        img = fill.get((ds, 'image_feat.npy'), ('', ''))
        txt = fill.get((ds, 'text_feat.npy'), ('', ''))
        w.writerow([ds, len(users), len(items), n, f'{n / (len(users) * len(items)):.6f}', split['0'], split['1'],
                    split['2'], img[0], txt[0], img[1], npy_shape(os.path.join(DATA, ds, 'image_feat.npy'))[1],
                    npy_shape(os.path.join(DATA, ds, 'text_feat.npy'))[1],
                    npy_shape(os.path.join(DATA, ds, 'video_feat.npy'))[1] if os.path.exists(os.path.join(DATA, ds, 'video_feat.npy')) else '',
                    len(ntest), sum(1 for v in ntest.values() if v == 1)])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', 'python3', '-', 'scan'], stdin=fh,
                             capture_output=True, check=True)
    with open(os.path.join(os.path.dirname(here), 'dataset_stats.csv'), 'w', newline='', encoding='utf-8') as fh:
        fh.write(out.stdout.decode().replace('\r\n', '\n'))
    print(out.stdout.decode())


if __name__ == '__main__':
    scan() if len(sys.argv) > 1 and sys.argv[1] == 'scan' else main()
