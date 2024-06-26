# Notes

- Python dependencies are outlined in the `pyproject.toml` file.
- Specific versions of those dependencies are specified in `pdm.lock` -- this is an autogenerated file that you should never need to edit manually.

# Installation

1. `cd` into the root of the `uncertainty` repository.

2. Delete the previously created virtual environment, if it exists.
   ```bash
   rm -rf .venv
   ```

3. Create a Python virtual environment.
   ```bash
   mkdir -p .venv
   python3.10 -m venv .venv --prompt=uncertainty
   ```
4. Activate the virtual environment.
   ```bash
   source .venv/bin/activate
   ```
   Potentially add this command to your `.bashrc` so you don't have to run this every time you open a new terminal.

5. Install [pdm](https://pdm-project.org/en/latest/) with
   ```bash
   pip install pdm
   ```

6. Install Python dependencies via [pdm](https://pdm-project.org/latest/).
   ```bash
   pdm install
   ```

# Adding new dependencies

1. Say you want to add the dependency `pandas`, run
   ```bash
   pdm add pandas
   ```
   This changes both the `pyproject.toml` and `pdm.lock` files.

2. Commit both the changed files!
   ```bash
   git add pyproject.toml pdm.lock
   git commit -m "Update pdm dependencies"
   ```

3. Git push!
   ```bash
   git push
   ```