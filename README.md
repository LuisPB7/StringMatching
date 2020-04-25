# StringMatching

First, please download the datasets folder from the following link: https://mega.nz/#!9gYz0abK!Jo0ouFJ6i-KKTWTqA-DKQcmlDFVXt0pCGO-do56gKj0

Requires ```pytorch``` and ```pytorch-lightning```

Usage:

```python3 main.py --model=MODEL --train=TRAIN --test=TEST```

where 

MODEL in ```{'rs','rs_pool', 'rs_pentanh','rs_pentanh_pool', 'rs_hardatt', 'rs_pentanh_hardatt', 'transformer', 'r_transformer', 'mogrifier_lstm', 'r_mogrifier_transformer', 'transformer_interaction', 'sha_rnn' }```

TRAIN/TEST in ```{'geonames', 'geonames_1', 'geonames_2', 'persons', 'persons_1', 'persons_2', 'organizations', 'organizations_1', 'organizations_2'}```

Results will be in a generated (or appended to the end of a) ```results.txt``` file.
