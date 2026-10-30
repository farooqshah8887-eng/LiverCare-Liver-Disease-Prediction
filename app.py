
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import pandas as pd
from pathlib import Path

# -----------------------------------
# 1. Flask Configuration
# -----------------------------------

app = Flask(__name__)
app.secret_key = "livercare-development-secret-key"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "livercare.db"
MODEL_PATH = BASE_DIR / "liver_model.pkl"


# -----------------------------------
# 2. Load Machine Learning Model
# -----------------------------------

model = None

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully!")

except Exception as error:
    print("Model loading error:", error)


# -----------------------------------
# 3. Database Connection
# -----------------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_patient_id_column(conn):
    columns = conn.execute(
        "PRAGMA table_info(patients)"
    ).fetchall()

    column_names = [column["name"] for column in columns]

    if "patient_id" in column_names:
        return "patient_id"

    if "id" in column_names:
        return "id"

    raise RuntimeError("Patient ID column not found.")


# -----------------------------------
# 4. Initialize Database
# -----------------------------------

def initialize_database():

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS lab_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            prediction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# -----------------------------------
# 5. Home Page
# -----------------------------------

@app.route("/")
def home():
    return redirect(url_for("register"))


# -----------------------------------
# 6. Patient Registration
# -----------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "GET":
        return render_template("register.html")

    conn = None

    try:
        patient_name = request.form.get(
            "patient_name", ""
        ).strip()

        age = int(request.form.get("age", "0"))

        gender = request.form.get(
            "gender", ""
        ).strip()

        if not patient_name:
            return "Patient name is required.", 400

        if age < 1 or age > 120:
            return "Please enter a valid age.", 400

        if gender not in ["Male", "Female"]:
            return "Please select a valid gender.", 400

        conn = get_db_connection()

        cursor = conn.execute("""
            INSERT INTO patients
            (patient_name, age, gender)
            VALUES (?, ?, ?)
        """, (patient_name, age, gender))

        patient_id = cursor.lastrowid

        conn.commit()

        session["patient_id"] = patient_id

        return redirect(
            url_for(
                "lab_assessment",
                patient_id=patient_id
            )
        )

    except ValueError:
        return "Please enter a valid age.", 400

    except Exception:
        app.logger.exception("Registration failed")
        return "Registration failed. Check terminal.", 500

    finally:
        if conn is not None:
            conn.close()


# -----------------------------------
# 7. Lab Assessment Page
# -----------------------------------

@app.route("/assessment/<int:patient_id>")
def lab_assessment(patient_id):

    conn = None

    try:
        conn = get_db_connection()

        id_column = get_patient_id_column(conn)

        if id_column == "id":
            query = "SELECT * FROM patients WHERE id = ?"
        else:
            query = "SELECT * FROM patients WHERE patient_id = ?"

        patient = conn.execute(
            query, (patient_id,)
        ).fetchone()

        if patient is None:
            return "Patient not found.", 404

        session["patient_id"] = patient_id

        return render_template(
            "assessment.html",
            patient=patient,
            patient_id=patient_id
        )

    except Exception:
        app.logger.exception("Assessment page failed")
        return "Could not open assessment page.", 500

    finally:
        if conn is not None:
            conn.close()


# -----------------------------------
# 8. Liver Disease Prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return "ML model could not be loaded.", 500

    conn = None

    try:
        patient_id = request.form.get(
            "patient_id", type=int
        )

        if patient_id is None:
            patient_id = session.get("patient_id")

        if patient_id is None:
            return redirect(url_for("register"))

        # Read laboratory values

        patient_data = {
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Total_Bilirubin": float(
                request.form["Total_Bilirubin"]
            ),
            "Direct_Bilirubin": float(
                request.form["Direct_Bilirubin"]
            ),
            "Alkaline_Phosphotase": float(
                request.form["Alkaline_Phosphotase"]
            ),
            "Alamine_Aminotransferase": float(
                request.form["Alamine_Aminotransferase"]
            ),
            "Aspartate_Aminotransferase": float(
                request.form["Aspartate_Aminotransferase"]
            ),
            "Total_Proteins": float(
                request.form["Total_Proteins"]
            ),
            "Albumin": float(
                request.form["Albumin"]
            ),
            "Albumin_and_Globulin_Ratio": float(
                request.form["Albumin_and_Globulin_Ratio"]
            )
        }

        # Feature Engineering

        total_bilirubin = patient_data["Total_Bilirubin"]
        direct_bilirubin = patient_data["Direct_Bilirubin"]
        total_proteins = patient_data["Total_Proteins"]
        albumin = patient_data["Albumin"]

        ast = patient_data["Aspartate_Aminotransferase"]
        alt = patient_data["Alamine_Aminotransferase"]

        patient_data["Bilirubin_Ratio"] = (
            total_bilirubin / direct_bilirubin
            if direct_bilirubin != 0 else 0
        )

        patient_data["Albumin_Protein_Ratio"] = (
            albumin / total_proteins
            if total_proteins != 0 else 0
        )

        patient_data["AST_ALT_Ratio"] = (
            ast / alt if alt != 0 else 0
        )

        # Convert into DataFrame

        patient_df = pd.DataFrame([patient_data])

        # Match model feature order

        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

            missing_features = [
                feature for feature in expected_features
                if feature not in patient_df.columns
            ]

            if missing_features:
                raise ValueError(
                    "Missing model features: " + str(missing_features)
                )

            patient_df = patient_df[expected_features]

        # Model Prediction

        prediction = model.predict(patient_df)[0]

        if str(prediction) == "1":
            result = "Liver disease risk indicated by the model"
        else:
            result = "Liver disease risk not indicated by the model"

        # Save prediction

        conn = get_db_connection()

        id_column = get_patient_id_column(conn)

        if id_column == "id":
            query = "SELECT * FROM patients WHERE id = ?"
        else:
            query = "SELECT * FROM patients WHERE patient_id = ?"

        patient = conn.execute(
            query, (patient_id,)
        ).fetchone()

        if patient is None:
            return "Patient not found.", 404

        conn.execute("""
            INSERT INTO lab_assessments
            (patient_id, prediction)
            VALUES (?, ?)
        """, (patient_id, result))

        conn.commit()

        total_patients = conn.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0]

        return render_template(
            "dashboard.html",
            prediction=result,
            patient_id=patient_id,
            patient_name=patient["patient_name"],
            total_patients=total_patients
        )

    except Exception as error:
        app.logger.exception("Prediction failed")

        return (
            "Prediction error: " + str(error),
            400
        )

    finally:
        if conn is not None:
            conn.close()


# -----------------------------------
# 9. Dashboard
# -----------------------------------

@app.route("/dashboard")
def dashboard():

    conn = None

    try:
        conn = get_db_connection()

        total_patients = conn.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0]

        patient_id = request.args.get(
            "patient_id",
            default=session.get("patient_id"),
            type=int
        )

        return render_template(
            "dashboard.html",
            total_patients=total_patients,
            patient_id=patient_id
        )

    except Exception:
        app.logger.exception("Dashboard failed")
        return "Dashboard loading failed.", 500

    finally:
        if conn is not None:
            conn.close()


# -----------------------------------
# 10. Run Flask Application
# -----------------------------------
# All Registered Patients

@app.route("/patients")
def all_patients():
    conn = None

    try:
        conn = get_db_connection()
        id_column = get_patient_id_column(conn)

        patients = conn.execute(
            f"SELECT * FROM patients ORDER BY {id_column} DESC"
        ).fetchall()

        return render_template(
            "patients.html",
            patients=patients,
            id_column=id_column
        )

    except Exception:
        app.logger.exception("Patients page failed")
        return "Could not load patients.", 500

    finally:
        if conn is not None:
            conn.close()
if __name__ == "__main__":
    app.run(debug=True)
