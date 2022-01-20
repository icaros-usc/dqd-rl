"""Entry point for running all experiments."""
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import fire
import gin
import torch
from dask.distributed import Client
from logdir import LogDir

from src.manager import Manager
from src.me_es.run import me_es_main
from src.utils.logging import setup_logging
from src.utils.worker_state import init_noise_table


def load_gin_config(config_file: str):
    """Loads gin configuration file.

    If the name is of the form X_test, both X and config/test.gin are loaded.
    """
    if config_file.endswith("_test"):
        config_file = config_file[:-len("_test")]
        is_test = True
    else:
        is_test = False

    # config/test.gin would need to be placed within src/ if main.py is not run
    # from root dir of repo.
    gin.parse_config_file(config_file)
    if is_test:
        gin.parse_config_file("config/test.gin")

        # Append " Test" to the experiment name.
        gin.bind_parameter("experiment.name",
                           gin.query_parameter("experiment.name") + " Test")

    gin.finalize()


def check_env():
    """Environment check(s)."""
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1', \
        ("OPENBLAS_NUM_THREADS must be set to 1 so that the numpy in each "
         "worker does not throttle each other. If you are running in the "
         "Singularity container, this should already be set.")


def setup_logdir(seed: int,
                 slurm_logdir: Union[str, Path],
                 reload_dir: Optional[str] = None):
    """Creates the logging directory with a LogDir object.

    Args:
        seed: Master seed.
        slurm_logdir: Directory for storing Slurm logs. Pass None if not
            applicable.
        reload_dir: Directory for reloading. If passed in, this directory will
            be reused as the logdir.
    """
    name = gin.query_parameter("experiment.name")

    if reload_dir is not None:
        # Reuse existing logdir.
        reload_dir = Path(reload_dir)
        logdir = LogDir(name, custom_dir=reload_dir)
    else:
        # Create new logdir.
        logdir = LogDir(name, rootdir="./logs")

    # Save configuration options.
    with logdir.pfile("config.gin").open("w") as file:
        file.write(gin.config_str(max_line_length=120))

    # Write a README.
    logdir.readme(git_commit=True, info=[f"Seed: {seed}"])

    # Write the seed.
    with logdir.pfile("seed").open("w") as file:
        file.write(str(seed))

    if slurm_logdir is not None:
        # Write the logging directory to the slurm logdir.
        with (Path(slurm_logdir) / "logdir").open("w") as file:
            file.write(str(logdir.logdir))

        # Copy the hpc config.
        hpc_config = glob.glob(str(Path(slurm_logdir) / "config" / "*.sh"))[0]
        hpc_config_copy = logdir.file("hpc_config.sh")
        shutil.copy(hpc_config, hpc_config_copy)

    return logdir


@gin.configurable(denylist=["client", "logdir"])
def experiment(client: Client,
               logdir: LogDir,
               seed: int,
               reload: bool = False,
               name: str = gin.REQUIRED,
               use_me_es: bool = False):
    """Executes a distributed experiment on Dask.

    Args:
        client: A Dask client for running distributed tasks.
        logdir: A logging directory instance for recording info.
        seed: Master seed for the experiment.
        reload: Whether to reload experiment from logdir.
        name: Name of the experiment.
        use_me_es: Whether to use ME-ES code.
    """
    logging.info("Experiment Name: %s", name)

    if use_me_es:
        client.register_worker_callbacks(init_noise_table)
        str_logdir = str(logdir.logdir) + "/"  # me_es code assumes a "/"
        me_es_main(
            client=client,
            log_dir=str_logdir,
            seed=seed,
        )
    else:
        # All further configuration to Manager is handled by gin.
        Manager(
            client=client,
            logdir=logdir,
            seed=seed,
            reload=reload,
        ).execute()


def main(
    config: str,
    seed: int,
    address: str = "127.0.0.1:8786",
    reload_dir: str = None,
    slurm_logdir=None,
):
    """Parses command line flags and sets up and runs experiment.

    Args:
        config: GIN configuration file. To pass a test config for `X`, pass in
            `X_test`. Then, `X` and `config/test.gin` will be included.
        seed: Master seed.
        address: Dask scheduler address.
        reload_dir: Path to previous logging directory for reloading the
            algorithm. New logs are also stored in this directory.
        slurm_logdir: Directory storing slurm output.
    """
    load_gin_config(config)
    check_env()

    logdir = setup_logdir(seed, slurm_logdir, reload_dir)

    client = Client(address)

    setup_logging(on_worker=False)
    client.register_worker_callbacks(setup_logging)

    # On the workers, PyTorch is entirely CPU-based. Since we run multiple
    # processes on each cluster node, allowing PyTorch to be multithreaded would
    # result in race conditions and thus slow down the code. This is similar to
    # how we force numpy and OpenBLAS to be single-threaded.
    client.register_worker_callbacks(lambda: torch.set_num_threads(1))

    # We wait for at least one worker to join the cluster before doing anything,
    # as methods like client.scatter fail when there are no workers.
    logging.info("Waiting for at least 1 worker to join cluster")
    client.wait_for_workers(1)
    logging.info("At least one worker has joined")

    logdir.save_data(client.ncores(), "client.json")
    logging.info("Master Seed: %d", seed)
    logging.info("Logging Directory: %s", logdir.logdir)
    logging.info("CPUs: %s", client.ncores())
    logging.info("===== Config: =====\n%s", gin.config_str())

    # PyTorch seeding is tricky. However, seeding here should be enough because
    # we only use PyTorch randomness in the initialization of the network. If we
    # use randomness during the iterations, reloading will not be "correct"
    # since we would be re-seeding at a generation other than the first. See
    # here for more info: https://pytorch.org/docs/stable/notes/randomness.html
    # By the way, we add 42 in order to avoid using the same seed as other
    # places.
    torch.manual_seed(seed + 42)

    experiment(
        client=client,
        logdir=logdir,
        seed=seed,
        reload=reload_dir is not None,
    )


if __name__ == "__main__":
    fire.Fire(main)
