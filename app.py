import os
import sys
import re
from datetime import datetime
from doctor_service import search_doctors, search_pharmacies
from map_builder import build_doctor_map
import pandas as pd
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from src.logger import logging
from src.exception import CustomException
from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)
app = Flask(__name__)
app.secret_key = 'super_secret_medico_key'

# Database Configuration
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    db_url = "sqlite:///medico.db"
else:
    # Standardize URL: strip whitespace and fix common protocol issues
    db_url = db_url.strip().strip("'\"")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('artifacts', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    records = db.relationship('MedicalRecord', backref='user', lazy=True)

class MedicalRecord(db.Model):
    __tablename__ = 'medical_records'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    record_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── HOME ROUTE ─────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    logging.info("Home page accessed")
    return render_template("index.html")


# ── PREDICT ROUTE ───────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return render_template("home.html")

    try:
        logging.info("Prediction request received")

        # ── Get form data ──────────────────────
        data = CustomData(
            symptoms  = request.form.get("symptoms"),
            age_group = request.form.get("age_group"),
            severity  = request.form.get("severity"),
            allergies = request.form.get("allergies")
        )

        # ── Convert to DataFrame ───────────────
        df = data.get_data_as_dataframe()
        logging.info(f"Input DataFrame:\n{df}")

        # ── Run Prediction ─────────────────────
        pipeline = PredictPipeline()
        result   = pipeline.predict(df)
        details  = pipeline.get_medicine_details(
            result["medicine"]
        )

        logging.info(f"Prediction result: {result}")
        
        emergency_alert = False
        severity = request.form.get("severity", "")
        if severity.lower() == "severe":
            emergency_alert = True

        return render_template(
            "home.html",
            result    = result,
            details   = details,
            symptoms  = request.form.get("symptoms"),
            age_group = request.form.get("age_group"),
            severity  = request.form.get("severity"),
            allergies = request.form.get("allergies"),
            emergency_alert = emergency_alert
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


# ── API ROUTE (JSON) ────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        # Get JSON data
        body = request.get_json()

        data = CustomData(
            symptoms  = body.get("symptoms"),
            age_group = body.get("age_group", "Adult"),
            severity  = body.get("severity",  "Mild"),
            allergies = body.get("allergies", "None")
        )

        df       = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result   = pipeline.predict(df)
        details  = pipeline.get_medicine_details(
            result["medicine"]
        )
        
        emergency_alert = False
        if body.get("severity", "").lower() == "severe":
            emergency_alert = True

        return jsonify({
            "success"   : True,
            "medicine"  : result["medicine"],
            "specialty" : result["specialty"],
            "confidence": result["confidence"],
            "details"   : details,
            "emergency_alert": emergency_alert

        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error"  : str(e)
        }), 500
    

# ── DOCTOR SEARCH ROUTE ─────────────────────────
# ── DOCTOR SEARCH ROUTE ─────────────────────────
@app.route("/doctors", methods=["GET", "POST"])
def doctors():
    if request.method == "GET":
        return render_template("doctor.html")

    try:
        specialty = request.form.get("specialty")
        location  = request.form.get("location")

        # Optional: replace radius with limit
        ##limit = int(request.form.get("limit") or 5)
        

        logging.info(f"Searching {specialty} near {location}")

        # ✅ Call updated function
        doctor_list = search_doctors(specialty, location, )

        # ✅ Build map
        map_html = build_doctor_map(doctor_list)

        return render_template(
            "doctor.html",
            doctors   = doctor_list,
            map_html  = map_html,
            specialty = specialty,
            location  = location
        )

    except Exception as e:
        raise CustomException(e, sys)

# ── PHARMACY SEARCH ROUTE ─────────────────────────
@app.route("/pharmacies", methods=["GET", "POST"])
def pharmacies():
    if request.method == "GET":
        return render_template("pharmacy.html")

    try:
        location  = request.form.get("location")
        logging.info(f"Searching pharmacies near {location}")

        pharmacy_list = search_pharmacies(location)
        map_html = build_doctor_map(pharmacy_list) # Reusing map builder

        return render_template(
            "pharmacy.html",
            pharmacies = pharmacy_list,
            map_html  = map_html,
            location  = location
        )
    except Exception as e:
        raise CustomException(e, sys)

# ── AUTHENTICATION ROUTES ───────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('register'))
            
        new_user = User(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('predict'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
            
        login_user(user)
        return redirect(url_for('predict'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ── DASHBOARD (MEDICAL RECORDS) ─────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        record_type = request.form.get('record_type')
        description = request.form.get('description')
        file = request.files.get('file')
        
        file_path = None
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
        new_record = MedicalRecord(
            user_id=current_user.id,
            record_type=record_type,
            description=description,
            file_path=file_path
        )
        db.session.add(new_record)
        db.session.commit()
        flash('Record added successfully')
        return redirect(url_for('dashboard'))
        
    records = MedicalRecord.query.filter_by(user_id=current_user.id).order_by(MedicalRecord.date_added.desc()).all()
    return render_template('dashboard.html', user=current_user, records=records)

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view_record/<int:record_id>')
@login_required
def view_record(record_id):
    record = MedicalRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    if not record.file_path:
        flash('No file associated with this record')
        return redirect(url_for('dashboard'))
        
    filename = os.path.basename(record.file_path)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/edit_record/<int:record_id>', methods=['POST'])
@login_required
def edit_record(record_id):
    record = MedicalRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    record.record_type = request.form.get('record_type', record.record_type)
    record.description = request.form.get('description', record.description)
    
    file = request.files.get('file')
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        record.file_path = file_path
        
    db.session.commit()
    flash('Record updated successfully')
    return redirect(url_for('dashboard'))

@app.route('/delete_record/<int:record_id>')
@login_required
def delete_record(record_id):
    record = MedicalRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    db.session.delete(record)
    db.session.commit()
    flash('Record deleted')
    return redirect(url_for('dashboard'))


# ── CHATBOT ROUTE ───────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        body = request.get_json()
        if not body or not body.get("message"):
            return jsonify({"success": False, "error": "No message provided"}), 400

        user_msg = body["message"].strip().lower()
        reply = _chatbot_reply(user_msg)

        return jsonify({
            "success": True,
            "reply": reply,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Chatbot error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _chatbot_reply(msg: str) -> str:
    """Rule-based medical chatbot logic."""
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(g in msg for g in greetings):
        return ("👋 Hello! I'm **MediBot**, your AI health assistant.\n\n"
                "I can help you with:\n"
                "• 🩺 Understanding symptoms\n"
                "• 💊 General medicine info\n"
                "• 🏥 Finding a specialist\n"
                "• 🚨 Emergency guidance\n\n"
                "What health question can I help you with today?")

    if any(w in msg for w in ["bye", "goodbye", "thanks", "thank you"]):
        return "Take care and stay healthy! 💚 Don't hesitate to come back if you have more health questions."

    emergency_keywords = ["chest pain", "heart attack", "stroke", "can't breathe",
                          "unconscious", "severe bleeding", "overdose", "seizure", "emergency"]
    if any(k in msg for k in emergency_keywords):
        return ("🚨 **This sounds like a medical emergency!**\n\n"
                "**Please call emergency services (112 / 108 / 911) immediately.**\n\n"
                "**Go to the nearest hospital or call 108 NOW.**")

    if re.search(r"fever|high temperature|chills|pyrexia", msg):
        return ("🌡️ **Fever Guidance**\n\n"
                "• Rest and drink plenty of fluids\n"
                "• Paracetamol (500mg every 4–6 hours) for adults\n"
                "• See a doctor if fever > 39.5°C or lasts more than 3 days")

    if re.search(r"headache|migraine|head pain", msg):
        return ("🧠 **Headache Guidance**\n\n"
                "• Rest in a quiet, dark room\n"
                "• Stay hydrated\n"
                "• Paracetamol or Ibuprofen (if no allergies)\n"
                "• Seek urgent care for sudden severe headache")

    if re.search(r"cold|flu|cough|runny nose|sore throat|sneezing", msg):
        return ("🤧 **Cold & Flu Guidance**\n\n"
                "• Rest and drink warm fluids\n"
                "• Steam inhalation for congestion\n"
                "• Paracetamol for fever/body aches\n"
                "• See a doctor if symptoms worsen after 5 days")

    if re.search(r"stomach|nausea|vomit|diarrhea|acidity|heartburn", msg):
        return ("🫃 **Digestive Issue Guidance**\n\n"
                "• Sip clear fluids slowly\n"
                "• ORS for diarrhea\n"
                "• Antacids for acidity/heartburn\n"
                "• Seek care if symptoms last > 3 days")

    if re.search(r"doctor|specialist|find|hospital|near me", msg):
        return ("🏥 **Finding a Doctor**\n\n"
                "Use our [Doctor Finder](/doctors) to search for specialists near you!\n\n"
                "Enter a specialty (e.g. Cardiologist) and your city to see results on a map.")

    return ("🤔 I'm not sure I fully understand that. Could you rephrase?\n\n"
            "You can ask me about: fever, headache, cold, stomach issues, or finding a doctor nearby.")

# ── RUN ────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host  = "0.0.0.0",
        port  = port,
        debug = True
    )
