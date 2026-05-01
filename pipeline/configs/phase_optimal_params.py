
# Optimal hyperparameters found from 150-trial SBTG tuning (Jan 2026)
# Model type 'regime_gated' was selected for all phases.

PHASE_OPTIMAL_PARAMS = {
    "baseline": {
        "dsm_lr": 0.0004916455477666916,
        "dsm_epochs": 158,
        "dsm_noise_std": 0.4999196192069903,
        "dsm_hidden_dim": 128,
        "structured_hidden_dim": 64,
        "structured_l1_lambda": 0.0001766488143459574,
        "fdr_alpha": 0.1,
        "model_type": "regime_gated",
        "num_regimes": 3
    },
    "butanone": {
        "dsm_lr": 0.0003148584274509215,
        "dsm_epochs": 114,
        "dsm_noise_std": 0.49959533295283315,
        "dsm_hidden_dim": 128,
        "structured_hidden_dim": 64,
        "structured_l1_lambda": 0.002218772569946514,
        "fdr_alpha": 0.15,
        "model_type": "regime_gated",
        "num_regimes": 3
    },
    "pentanedione": {
        "dsm_lr": 0.00038752472239695496,
        "dsm_epochs": 100,
        "dsm_noise_std": 0.49910527892779116,
        "dsm_hidden_dim": 128,
        "structured_hidden_dim": 64,
        "structured_l1_lambda": 0.006939121826408764,
        "fdr_alpha": 0.1,
        "model_type": "regime_gated",
        "num_regimes": 2
    },
    "nacl": {
        "dsm_lr": 0.00016276211024372048,
        "dsm_epochs": 220,
        "dsm_noise_std": 0.49986445625473663,
        "dsm_hidden_dim": 128,
        "structured_hidden_dim": 64,
        "structured_l1_lambda": 0.0014529207908379326,
        "fdr_alpha": 0.2,
        "model_type": "regime_gated",
        "num_regimes": 2
    }
}
