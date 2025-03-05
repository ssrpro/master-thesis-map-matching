# Setup

Make sure to setup venv before installing packages or running any code!

Setup virtual environment:
```console
$ python -m venv .venv
```
Activate virutal environment:
```console
$ source .venv/Scripts/activate
````

Check for correct path:
Activate virutal environment:
```console
$ which python
````

Install packages
```console
$ pip install numba pytest ruff pyright jupyter numpy
```

Run the Jupyter notebooks in the notebook folder.

# Run
Run all tests:
```console
$ pytest
```

Run linter:
```console
$ ruff check
```
