python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Expected data format (BYOD)

train.csv: object_id, <features...>, type — the target column is called "type" (if it's named differently, the script will try to find it automatically).

test.csv: object_id, <features...>

sample_submission.csv: object_id,<class1>,...,<classN> — the class columns must match your labels.

The script will automatically calculate the derived features (add_features()), align the train/test columns, and process basic physics heuristics (parallax, proper motion, etc., if available).

Run
Training (get local weights and predictions)
python main.py \
--train path/to/train.csv \
--test path/to/test.csv \
--sample path/to/sample_submission.csv \
--out predictions.csv \
--save_models \
--device gpu # or cpu

Inference on your own trained weights
python main.py \
--train path/to/train.csv \
--test path/to/test.csv \
--sample path/to/sample_submission.csv \
--out predictions.csv \
--load_models \
--device gpu # or cpu

Parameters:

--save_models — save the final full-data CatBoost model and metalayer (to the local models/ folder, which you do not commit).

--load_models — load locally saved weights (models/model_full.cbm, models/meta_lr.joblib) and use them.

--blend_final — blending weight of the final full-data model with the K×seeds ensemble (default 0.5).

--device — gpu (default) or cpu. If you don't have CUDA, specify --device cpu.

Notes

Macro-F1 is used as the target metric in validation.

If you don't have parallax, pm_ra/pm_dec, or background_noise features, the script will continue to run correctly, but some of the physical rules will be disabled.

For robustness on rare classes, class weights and segment bias weights (global / extragal / stellar), selected by Optuna, are used.

Optuna logs are printed to stdout by default (you can wrap the call with a separate logger/output redirector)
