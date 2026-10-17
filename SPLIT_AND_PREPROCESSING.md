# Datasets, splits, and preprocessing

This file specifies the data that every number of the manuscript is computed on. The raw interaction and feature files are
not redistributed here; they come from the public releases named below.

## Datasets

| Dataset | Source | Users | Items | Interactions | Train | Valid | Test | Item features (dimensions) | Items with a filled image vector |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Baby | MMRec (Amazon Baby) | 19,445 | 7,050 | 160,792 | 118,551 | 20,559 | 21,682 | image 4,096, text 384 | 0 |
| Sports | MMRec (Amazon Sports and Outdoors) | 35,598 | 18,357 | 296,337 | 218,409 | 37,899 | 40,029 | image 4,096, text 384 | 0 |
| Clothing | MMRec (Amazon Clothing) | 39,387 | 23,033 | 278,677 | 197,338 | 40,150 | 41,189 | image 4,096, text 384 | 0 |
| Pantry | MISSRec (Amazon Prime Pantry) | 13,101 | 4,898 | 126,797 | 100,602 | 13,094 | 13,101 | CLIP-B/32 image 512, text 512 | 264 |
| Scientific | MISSRec (Amazon Industrial and Scientific) | 8,442 | 4,385 | 59,136 | 42,268 | 8,426 | 8,442 | CLIP-B/32 image 512, text 512 | 1,438 |
| Instruments | MISSRec (Amazon Musical Instruments) | 24,962 | 9,964 | 207,051 | 157,171 | 24,918 | 24,962 | CLIP-B/32 image 512, text 512 | 2,565 |
| Arts | MISSRec (Amazon Arts, Crafts and Sewing) | 45,486 | 21,019 | 392,366 | 301,467 | 45,413 | 45,486 | CLIP-B/32 image 512, text 512 | 8,595 |
| Office | MISSRec (Amazon Office Products) | 87,346 | 25,986 | 679,508 | 504,965 | 87,197 | 87,346 | CLIP-B/32 image 512, text 512 | 6,724 |
| MicroLens | MicroLens | 98,129 | 17,228 | 705,174 | 500,064 | 101,121 | 103,989 | image 1,024, text 1,024, video 768 | 0 |

The counts are those of Table S1 of the supplementary material and of `tables/src/collect_dataset_stats.py` (output
`tables/dataset_stats.csv`, not part of this staging folder). Baby, Pantry, Scientific, and Sports are the four datasets of
the main comparison; the analyses of the fused baseline F, the classical fill C, and the headroom N' add the other five.

## Split

- Each dataset keeps the split of its source files. The interaction file `<dataset>.inter` (tab separated, columns
  `userID`, `itemID`, `x_label`) marks every interaction as training (`x_label = 0`), validation (`1`), or test (`2`).
- All models, classical rankers, and fusion steps read the same interaction and feature files of a dataset.
- The five MISSRec domains hold exactly one test interaction per user (the Test column equals the number of users).
- Hyperparameters, fusion weights, and the settings of the classical rankers are chosen on validation Recall@20 only;
  the test split is read only for the reported metrics.

## Features and their preprocessing

- Item features are frozen (no fine-tuning). Baby, Sports, and Clothing use the 4,096-d image and 384-d text features
  distributed with MMRec. The five MISSRec domains use 512-d CLIP-B/32 image and text features. MicroLens adds 768-d video
  features to its image and text features.
- An item whose image feature vector is all zeros gets the mean of the non-zero rows of that feature file. The last column
  of the table above counts these items. No text vector needed filling.
- The classical rankers use only the training interactions (and, for ItemKNN-CBF, the text features); see
  `server_code/` for RP3beta, ItemKNN-CBF, ItemKNN-CF, EASE, and their validation mix.

## Evaluation

- Full ranking over all items in the common MMRec-based pipeline; Recall@20 and NDCG@20.
- Every reported value is a mean over seeds 1-5 with one run per seed unless a caption states another seed count.
- Every significance mark uses one two-sided paired t-test over seeds with Holm correction over the datasets of a contrast
  (`tables/src/sigtest.py`). Against a deterministic classical ranker, the test reduces to a one-sample t-test.

## Item kNN cache builds (Baby)

The MES-F and MES-C runs on Baby use the second (v2) build of the item kNN cache, whereas the fused comparators use the
canonical build. `tables/src/data/baby_cache_audit.csv` and `tables/src/data/baby_cache_audit_g2m.csv` list the build of
every Baby run (Table S6 of the supplementary material).
