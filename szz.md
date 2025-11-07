python cold_start_dfp.py \
  --ratings_path ./ml-100k/u.data \
  --users_path   ./ml-100k/u.user \
  --n_clusters 8 \
  --min_support 0.15 \
  --theta 1.0 \
  --phi 0.001 \
  --ct 0.25 \
  --out_prefix movielens100k
