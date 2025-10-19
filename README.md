# Celestial Objects Classification (CatBoost + Optuna)

Classification of celestial objects (stars / quasars / galaxies / white dwarfs / red giants / exoplanet candidates) with rare-class stratification, TTA, and meta-blending.

> **Important:** The repository **does not** contain the dataset or pre-trained weights.
> You can train the model on your own data and obtain the weights locally.

## Features
- CatBoost MultiClass + custom stratification for rare classes.
- Optuna: joint hyperparameter and bias weight tuning (global/extragal/stellar) + temperature.
- Postprocessing with "physical" rules (soft/hard), TTA (photometric jitter).
- Meta-layer (LogisticRegression) based on OOF scores + physical features.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Ожидаемый формат данных (BYOD)

train.csv: object_id, <features...>, type — целевая колонка называется type (если называется иначе — скрипт попробует найти её автоматически).

test.csv: object_id, <features...>

sample_submission.csv: object_id,<class1>,...,<classN> — столбцы классов должны совпадать с вашими метками.

Скрипт сам досчитает производные признаки (add_features()), выровняет колонки train/test и обработает базовые физические эвристики (parallax, proper motion и т.п., если доступны).

Запуск
Обучение (получить локальные веса и предсказания)
python main.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --sample path/to/sample_submission.csv \
  --out predictions.csv \
  --save_models \
  --device gpu   # или cpu

Инференс на своих уже обученных весах
python main.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --sample path/to/sample_submission.csv \
  --out predictions.csv \
  --load_models \
  --device gpu   # или cpu


Параметры:

--save_models — сохранить финальную full-data CatBoost-модель и метаслой (в локальную папку models/, которую вы не коммитите).

--load_models — загрузить уже сохранённые локально веса (models/model_full.cbm, models/meta_lr.joblib) и использовать их.

--blend_final — вес блендинга финальной full-data модели с ансамблем K×seeds (по умолчанию 0.5).

--device — gpu (по умолчанию) или cpu. Если у вас нет CUDA — укажите --device cpu.



Примечания

Макро-F1 используется как целевая метрика в валидации.

Если у вас нет признаков parallax, pm_ra/pm_dec, background_noise, скрипт корректно продолжит работу, просто часть физ. правил будет отключена.

Для устойчивости на редких классах применяются классовые веса и сегментные bias-веса (global / extragal / stellar), которые подбираются Optuna.

Логи Optuna по умолчанию печатаются в stdout (можно обернуть вызов отдельным логгером/редиректором вывода).
