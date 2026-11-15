# AI vs Human Essay Detector

## Description of the Problem
Essays written by AI tools are hard to tell apart from human writing, which matters
for anyone who needs to check where a text came from.
This project builds a model that takes an essay and returns the probability that it
was written by AI. It could be used as a first screening step, not as proof.

## Dataset
Full dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text (about 1 GB, not included in this repo).
Download `AI_Human.csv` and put it in the project folder to run the notebook.
A 2% sample (`essays_sample.jsonl`) is included so the Docker image can train without it.

## Approach
1. **EDA**: class balance, essay length by class, most common words per class.
2. **Features**: TF-IDF on words and word pairs (30,000 features).
3. **Models**: Logistic Regression (tuned `C`) and XGBoost (tuned `eta`, `max_depth`,
   `min_child_weight`).
4. **Selection**: validation AUC for tuning, test set used once at the end.

5. ## Results
| Model | Validation AUC | Test AUC |
|---|---|---|
| Logistic Regression (C=10) | <0.99xx> | <0.99xx> |
| XGBoost (tuned) | <0.99xx> | <0.99xx> |

The final model is **Logistic Regression**: it scored as well as or better than XGBoost,
trains in seconds, and is easy to deploy.

## Files
- notebook.ipynb: EDA, model comparison and tuning
- train.py: trains the final pipeline and saves model.bin
- predict.py: Flask web service (/predict)
- Dockerfile, requirements.txt
- essays_sample.jsonl: sample used to train inside the Docker build

## How to run
docker build -t ai-essay-detector .
docker run -it --rm -p 9696:9696 ai-essay-detector

This trains the model during the build. Then, in another terminal:
python -c "import requests; print(requests.post('http://localhost:9696/predict', json={'text': 'Your essay here'}).json())"
