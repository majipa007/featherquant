# Approximation costs

Every rung of the ladder (spec §5) trades memory for quality. **No number
in this table may be a guess.** A row is `UNMEASURED` until a committed run
manifest in `bench/manifests/` produces it; the `source` column names that
manifest. The planner reads this file to populate its refusal message.

Measurement context for every PPL figure in this table: wikitext-2-raw
`wiki.test.raw`, context length 512, Qwen3 tokenizer. Peak Δ figures are
decimal MB (10^6 bytes, as file-size tools report); memory ceilings elsewhere
in this project are binary GiB.

| rung | flag | peak Δ | runtime Δ | PPL Δ | downstream task Δ | source |
|---|---|---|---|---|---|---|
| hessian_full | `--hessian-approx=full` | 0 | 0% | 0.00 | 0.00 | reference |
| hessian_blocked | `--hessian-approx=blocked` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| hessian_lowrank | `--hessian-approx=lowrank` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| hessian_diagonal | `--hessian-approx=diagonal` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_samples_64 | `--calib-samples=64` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_samples_32 | `--calib-samples=32` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_seqlen_256 | `--calib-seqlen=256` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_spill | `--calib-spill` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| stats_bf16 | `--stat-precision=bf16` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| kquant_group_joint | (implicit for K-quant targets) | 0 | 0% | UNMEASURED | UNMEASURED | UNMEASURED |
