# LiverCare — Liver Disease Prediction

Flask application for an educational machine-learning demonstration.

## Add your existing project files
- `app.py`
- `liver_model.pkl`
- `templates/` (all HTML files)
- `static/` (if present)
- These deployment files: `requirements.txt`, `Procfile`, `.gitignore`

Do not upload `livercare.db` or real patient information to GitHub.

## Local setup
```bash
python -m pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

## Render settings
Build command:
```bash
pip install -r requirements.txt
```
Start command:
```bash
gunicorn app:app
```
The Flask application object in `app.py` must be named `app`.

## Important
`scikit-learn==1.7.2` matches the version reported when your model was created. Pickle models can be incompatible across library versions.

A local SQLite database may not persist on an ephemeral host across restarts or redeploys. Configure persistent storage or a managed database before relying on saved patient records.

Use only dummy/demo patient data in a public demo. Model output is educational and is not a medical diagnosis.
