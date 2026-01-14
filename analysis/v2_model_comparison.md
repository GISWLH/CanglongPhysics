# V2 Model Comparison

This table compares the current V2.3 and V2.4 configs with a proposed V2.5 optimization (not implemented yet).

| Item | V2.3 (current) | V2.4 (current) | V2.5 (proposed) |
| --- | --- | --- | --- |
| Training script | `train_v2_3.py` | `train_v2_4.py` | TBD (new) |
| Model class | `canglong/model_v2_3.py` | `canglong/model_v2_4.py` | TBD (new) |
| embed_dim | 96 | 192 | 192 |
| num_heads | (8,16,16,8) | (8,16,16,8) | (12,24,24,12) |
| depths (L1/L2/L3/L4) | 2/6/6/2 | 2/6/6/2 | 4/8/8/4 |
| Wind Top-K | 4 | 3 | 4 (L2/L3 optional 2 or thresholded) |
| Wind-shift layers | L1 only | L1 only | L1 + L4 |
| window_size | (2,6,12) | (2,6,12) | (2,6,12) |
| wind_shift_scale | 2 | 2 | 2 |
| wind_speed_threshold | 0.5 | 0.5 | 0.5 |
| use_checkpoint | True | True | True |
| drop_path max | 0.2 | 0.2 | 0.3 |
| Optimizer / LR | Adam / 5e-4 | Adam / 5e-4 | Adam / 3e-4 |
| weight_decay | 0 | 0 | 1e-4 |
| Loss | MSE (surf + upper) | MSE (surf + upper) | MSE + key-variable weights + physical residuals (optional) |
| batch_size | 1 | 1 | 1 |
| AMP | True | True | True |
| Checkpoint interval | every 25 epochs | every 25 epochs | every 25 epochs |

Notes:
- V2.5 is a proposal and has not been implemented yet.
- V2.4 changes both embed_dim and Top-K, so it is not a pure width-only comparison.
