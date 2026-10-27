# Logs of the final runs

- `run_full_pipeline.log` — stage timestamps of the full BigFlow run (2026-10-09/10, 66.9M flows).
- `run_label_ambiguity.log`, `run_build_sample.log`, `run_part1.log`, `run_part5.log`, `run_robustA.log`, `run_robustB.log`, `run_egraphsage.log`, `run_figures.log` — the stages of that run.
- `run_part34.log` — parts 3 and 4 of that run. **Its Part 3 numbers for the 10-minute time-block split are superseded**: the block split was fixed on 2026-10-10 and Part 3 re-run, see `run_part3_blockfix.log` (the result CSVs contain the re-run). Part 4 is unaffected.
- `run_paper_tables.log` — table generation after the re-run.
- `run_unsw.log`, `run_ciciot.log` — external validation (2026-10-11).
- Logs of runs on an earlier 46-file subset of BigFlow are not included; their results are superseded.
