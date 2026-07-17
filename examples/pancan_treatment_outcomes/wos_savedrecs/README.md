This directory contains the Web of Science savedrecs text exports used by the
pancan treatment-outcomes example.

Run a small resumable batch from the repository root:

```bash
.venv/bin/python run_pipeline.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --wos examples/pancan_treatment_outcomes/wos_savedrecs \
  --output experiments/pancan/outputs \
  --limit 10
```

Use the same output directory for later runs. Completed papers are skipped, so
another `--limit 10` continues with the next unfinished papers.
