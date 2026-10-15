# Table S1b — hyper-parameter search space (read from `_code/da_reformer/hpo.py` and `models/baselines.py::suggest_hp` by recording the suggest calls)

30 Optuna trials per model (TPE sampler seed 2026, median pruner after 8 startup trials, fold-level), objective = mean A–D leave-one-well-out validation macro-F1 (protocol_v03.yaml `hpo`). '+VDL' models search the space of their base model; Mean pooling searches the ModernTCN space plus the waveform encoder. ModernTCN small_size is fixed at 5.

| Model | Parameter | Space |
|---|---|---|
| LSTM | lr | float [0.0001, 0.003] log |
| LSTM | weight_decay | float [1e-06, 0.01] log |
| LSTM | dropout | float [0, 0.3] |
| LSTM | hidden | {32, 64, 128} |
| LSTM | num_layers | int [1, 3] |
| GRU | lr | float [0.0001, 0.003] log |
| GRU | weight_decay | float [1e-06, 0.01] log |
| GRU | dropout | float [0, 0.3] |
| GRU | hidden | {32, 64, 128} |
| GRU | num_layers | int [1, 3] |
| Transformer | lr | float [0.0001, 0.003] log |
| Transformer | weight_decay | float [1e-06, 0.01] log |
| Transformer | dropout | float [0, 0.3] |
| Transformer | d_model | {32, 64, 128} |
| Transformer | n_heads | {2, 4, 8} |
| Transformer | e_layers | int [1, 3] |
| Transformer | d_ff | {64, 128, 256} |
| Reformer | lr | float [0.0001, 0.003] log |
| Reformer | weight_decay | float [1e-06, 0.01] log |
| Reformer | dropout | float [0, 0.3] |
| Reformer | d_model | {32, 64, 128} |
| Reformer | n_heads | {2, 4, 8} |
| Reformer | e_layers | int [1, 3] |
| Reformer | d_ff | {64, 128, 256} |
| Reformer | bucket_size | {4, 8, 16} |
| Reformer | n_hashes | {2, 4, 8} |
| Crossformer | lr | float [0.0001, 0.003] log |
| Crossformer | weight_decay | float [1e-06, 0.01] log |
| Crossformer | dropout | float [0, 0.3] |
| Crossformer | d_lift | {8, 16, 32} |
| Crossformer | d_model | {32, 64, 128} |
| Crossformer | n_heads | {2, 4, 8} |
| Crossformer | e_layers | int [1, 3] |
| Crossformer | d_ff | {64, 128, 256} |
| Crossformer | seg_len | {8, 12, 16, 24, 32} |
| Crossformer | factor | {1, 5, 10} |
| TimesNet | lr | float [0.0001, 0.003] log |
| TimesNet | weight_decay | float [1e-06, 0.01] log |
| TimesNet | dropout | float [0, 0.3] |
| TimesNet | d_model | {16, 32, 64} |
| TimesNet | d_ff | {16, 32, 64} |
| TimesNet | e_layers | int [1, 3] |
| TimesNet | top_k | int [2, 5] |
| TimesNet | num_kernels | {4, 6} |
| TimeMixer | lr | float [0.0001, 0.003] log |
| TimeMixer | weight_decay | float [1e-06, 0.01] log |
| TimeMixer | dropout | float [0, 0.3] |
| TimeMixer | d_lift | {8, 16} |
| TimeMixer | d_model | {16, 32} |
| TimeMixer | d_ff | {32, 64} |
| TimeMixer | e_layers | int [1, 3] |
| TimeMixer | down_sampling_layers | int [1, 3] |
| TimeMixer | moving_avg | {13, 25, 49} |
| ModernTCN | lr | float [0.0001, 0.003] log |
| ModernTCN | weight_decay | float [1e-06, 0.01] log |
| ModernTCN | dropout | float [0, 0.3] |
| ModernTCN | d_lift | {8, 16, 32} |
| ModernTCN | patch_size | {4, 8, 16} |
| ModernTCN | d_model | {32, 64} |
| ModernTCN | ffn_ratio | {1, 2} |
| ModernTCN | num_blocks | int [1, 2] |
| ModernTCN | large_size | {31, 51, 71} |
| Mean pooling | lr | float [0.0001, 0.003] log |
| Mean pooling | weight_decay | float [1e-06, 0.01] log |
| Mean pooling | dropout | float [0, 0.3] |
| Mean pooling | d_lift | {8, 16, 32} |
| Mean pooling | patch_size | {4, 8, 16} |
| Mean pooling | d_model | {32, 64} |
| Mean pooling | ffn_ratio | {1, 2} |
| Mean pooling | num_blocks | int [1, 2] |
| Mean pooling | large_size | {31, 51, 71} |
| Mean pooling | d_wave | {16, 32, 64} |
| Mean pooling | wave_channels | {16, 32, 64} |
