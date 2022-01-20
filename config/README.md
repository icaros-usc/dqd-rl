# Config

Configuration files are divided by experiment. In each directory, there is a
configuration file for each algorithm, as well as a `shared.gin` file with
configurations shared by the algorithms (typically these relate to the
environment / objective function or the archive).

There are some exceptions. `hpc/` contains configurations for HPC slurm nodes
(see `scripts/run_slurm.sh`), and `algorithms/` contains configurations for
various algorithms, which tend to be shared across experiments.

To write a new config, include:

- The `shared.gin` file for that directory
- An algorithm config from `algorithms/`
- The name of the experiment, `experiment.name`

To test a configuration in `main.py`, add `_test` to the end of a name, e.g.
`config/qd_ant/cma_mega_es.gin_test`. Then, the original config
(`config/qd_ant/cma_mega_es.gin`) and `config/test.gin` will be included.

As these configs are presented in pieces that import each other, it may be
difficult to understand them. To print out single config files without any
imports, use `src/dump_config.py`.
