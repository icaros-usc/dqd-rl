"""Prints out all configuration after loading a config file.

Since we import configs a lot, this can be helpful for seeing all the options
pulled in by a certain file.

Usage:
    python -m src.dump_config CONFIG
Example:
    python -m src.dump_config config/qd_ant/cma_mega_es.gin
"""
import fire
import gin

# Including this makes gin config work because main imports (pretty much)
# everything.
import src.main  # pylint: disable = unused-import


def dump_config(file):
    gin.parse_config_file(file)

    # Same settings as in main.py
    print(gin.config_str(max_line_length=120), end='')


if __name__ == "__main__":
    fire.Fire(dump_config)
