**Authors**: CARAYON Chloé - SPATZ Cécile (BD2)
**Date**: 21/11/2021
___
# Applications of Big Data
___

This project aims to apply some concepts and tools seen in the course Applications of Big Data at EFREI.
In order to run it, please install and configure poetry.

## Getting Start
---

- One can find the report [here](https://github.com/ChloeCarayon/application_bd/blob/develop/reports/CARAYON_SPATZ_Report.md). 


- We use poetry for our environment. 
One can use the already build environment OR re-install it:
``` 
make install
```
if issues, delete poetry.lock and re run the command above.


- To clean the code for production:
``` 
make check
```

- You need to configure your Kaggle API if you want to generate the datasets.
Follow the instructions below:
https://github.com/Kaggle/kaggle-api
If you encounter any issues, please verify your account.

###  Classical ML project
- Run project
``` 
make run
```

**Click** library allows you to select the action you want to do.

- H20 server 
http://127.0.0.1:54331
From the server, to import your H20 model Mojo format you can get them from models directory or mlruns directories.

- Sphinx documentation [here](https://github.com/ChloeCarayon/application_bd/blob/master/docs/build/html/index.html)

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

--- 


    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Sphinx project
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── visualisation      <- figures visualisation
    │
    ├── reports            <- Generated analysis as HTML, PDF, Markdown, etc.
    ├── pyproject.toml     <- defines the build system as a dependency.
    ├── poetry.lock        <- prevents you from automatically getting the latest versions of your dependencies.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- command line interfaces to choose action to perform
    │   ├── tracker.py     <- mlflow tracker   
    │   ├── utils.py       <- utils   
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
    │   │   └── model_parameters.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── do_explicability_h2o.py
    │       └── do_explicability_xboost.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
