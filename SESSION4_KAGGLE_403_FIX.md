# Session 4: Kaggle HTTP 403 and disk-space fix

Session 4 no longer auto-downloads Montgomery Lung because the official NLM
directory rejects Kaggle directory listing requests with HTTP 403. Montgomery
remains optional/manual in the data registry, but it is not used by automatic
sessions.

Session 4 now uses only direct official Simula archives:

- Primary comparison: `hyper_kvasir_seg`
- HC-only comparison: `kvasir_instrument`
- HC ablations: `kvasir_seg`

Checkpoint directories are deleted after evaluation when
`DELETE_CHECKPOINTS_AFTER_EVAL=1`, and downloaded archives are deleted after
preprocessing when `CLEAN_DOWNLOAD_ARCHIVES=1`.
