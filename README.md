# Applications of Big Data
==============================

This project aims to apply some concepts and tools seen in the course Applications of Big Data at EFREI.

## Getting Start

- Install dependencies
``` 
make install
```
if issues, delete poetry.lock and re run the command above.

- configure Kaggle API
Follow the instructions below:
https://github.com/Kaggle/kaggle-api
If you encounter any problems, please verify your account.

###  Classical ML project
- Run project
``` 
make run
```

###  MlFlow project

#### From src directory

```cd ``` to the ```src ``` directory.

- Run the MLfLow project with or without the parameters
```
 poetry run python tracker.py {ntrees} {max_depth} {learn_rate} {min_rows}
```
example:
``` 
poetry run python tracker.py 200 5 0.09 5
poetry run python tracker.py
```

- Comparing the models with MLflow UI
``` 
poetry run mlflow ui
```

-Running MlFlow server in local
``` 
poetry run mlflow models serve -m runs:/0a599c7b8fe6494eac8f2aeebb3d7b2e/model --port 1234 
```

#### From application_bd directory
```cd ..``` to the principal directory.

- Run to train model with default paramaters
```
poetry run mlflow run src --no-conda
```

- Run to train model with your paramaters
```
 poetry run mlflow run src --no-conda -P ntrees=250 -P max_depth=3 -P learn_rate=0.3 -P min_rows=8
```

- Specify server
```  poetry run mlflow server -m runs:/ac0a34dbf08345d3bc37563b6ae2b700/model --port 1234 --no-conda ```

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
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
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pyproject.toml     <- defines the build system as a dependency.
    ├── poetry.lock        <- prevents you from automatically getting the latest versions of your dependencies.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- command line interfaces to choose action to perform
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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>