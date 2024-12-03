# Project:

## Project Organization

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-sba-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

---

# Steps to follow after creating a project

## create and activate virtual environment

for Windows

```
python -m venv .venv
.venv/Scripts/activate
```

for Mac

```
python -m venv .venv
source .venv/bin/activate
```

## setup Git repository

### initialize repo

```
git init -b  main
```

### setup user and email

```
git config user.name "Stanislav Babenyshev"
git config user.email babenyshevs@gmail.com
```

### make initial commit

```
git add .
git commit -m "default initial commit"
```

### checkout develop branch (optional)

```
git checkout -b develop
```

### create remote repo on GitHub

discussed at [link](/guides/content/editing-an-existing-page%5D(https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github#adding-a-local-repository-to-github-using-git)

### add remote repository address to local one

```
git remote add origin REMOTE-URL
```

### push changes to remote

```
git push -u origin main
```

## install requirements

```
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## for NLP tasks install spacy model

the most straight forward way is

```
python -m spacy download de_core_news_lg
```

This can be although tedious because of the proxy settings.

Alternative approach is to download respective wheel of a [trained language pipeline](https://spacy.io/models/) and place into models folder, if model is large, add it to gitignore section. Finally, install the package.

```
pip install models/<model_name>
```

This approach is sometimes preferable, because one can control which version of model (s)he would have and keep this artifact, say, in Azure Blob, AWS S3 bucket, etc.

To see, which spacy models are downloaded and installed use:

```
python -m spacy info
```

To uninstall use $pip$:

```
pip uninstall en-core-web-sm
```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience