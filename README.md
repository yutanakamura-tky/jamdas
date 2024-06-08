# What's this?

This is a template repository customized for quick set up of research projects.

# Requirements

- Python => 3.8

# How to use this

1. Install Poetry:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

2. Change Config:

```sh
poetry config virtualenvs.in-project true
```

3. Fill the blank in `pyproject.toml` with your favorite project name: 

```sh
vi /path/to/project/pyproject.toml
```

```
(pyproject.toml)

[tool.poetry]
name = "" # REPLACE WITH YOUR FAVORITE PROJECT NAME
...
```

4. Update your project:

```sh
poetry update
```

5. Install packages using Poetry:

```sh
cd /path/to/project
poetry install
```

6. Install pre-commit hook:

```sh
cd /path/to/project
pre-commit install
```
