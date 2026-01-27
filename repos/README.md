# Machine Learning Repos

This markdown file describes the process of locating the desired ML projects and adding them.

## Locating Desired ML Projects

## Adding ML Projects

```
./scripts/import_repo.sh [url]
```

## If no requirements.txt

```
python3 -m venv venv
pip freeze > requirements.txt
source venv/bin/activate
pip install -r requirements.txt
```

## If requirements.txt exists

```
cd repos/[project]
source venv/bin/activate
pip install -r requirements.txt
```

## Deactivate Virtual Environment (venv)

```
deactivate
```

### notes on process

Created a script to import projects.
Then I have fixed code in project so it runs, e.g., csv path or removing GUI code.

diabetes_predictive_analysis: fixed standardization (scaler) because scaling was implemented but not applied due to a coding error

cars-classification: fixed nan values from dataset

Rock-or-Mine-prediction: fix numpy coding error & remove input prediction

Deep-Learning-Classification: fix Keras API shape formatting issue

CT_Project: fix pandas TypeError
