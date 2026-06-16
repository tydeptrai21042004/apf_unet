# Session 4: eight independent disk-safe runs

Each file is completely independent. It installs dependencies by default, prepares only its own dataset, runs only its assigned experiment, deletes checkpoints after evaluation, removes download archives/raw duplicates, and keeps only that part's metrics/results directory.

No part depends on another part. No result ZIP is created. The wrapper has no `all` mode, preventing accidental execution of the complete workload.

## Parts

1. `run_hc_session_4_part_1.sh`: HyperKvasir-SEG, Polyp-PVT + CaraNet
2. `run_hc_session_4_part_2.sh`: HyperKvasir-SEG, HSNet
3. `run_hc_session_4_part_3.sh`: HyperKvasir-SEG, CFANet
4. `run_hc_session_4_part_4.sh`: HyperKvasir-SEG, ResUNet++
5. `run_hc_session_4_part_5.sh`: Kvasir-Instrument, proposed HC model
6. `run_hc_session_4_part_6.sh`: Kvasir-SEG, HC kernel-5 ablation
7. `run_hc_session_4_part_7.sh`: Kvasir-SEG, identity-projection ablation
8. `run_hc_session_4_part_8.sh`: Kvasir-SEG, no-channel-expansion ablation

## Run one part

```bash
bash run_hc_session_4_part_1.sh
```

or:

```bash
bash run_hc_session_4.sh 1
```

Use a number from 1 through 8. Running without a number exits deliberately. Running all parts through the wrapper is deliberately unsupported.

## Defaults

```bash
INSTALL_DEPS=1
RUN_TESTS=0
DELETE_CHECKPOINTS_AFTER_EVAL=1
CLEAN_DOWNLOAD_ARCHIVES=1
CLEAN_PIP_CACHE=1
CLEAN_GROUP_OUTPUT_BEFORE_RUN=1
CLEAN_RAW_DATA_AFTER_RUN=1
```

Processed resized data and split files are retained so reruns can reuse them. Set `CLEAN_GROUP_OUTPUT_BEFORE_RUN=0` only when intentionally resuming an existing output directory.
