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
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from src.logger import logging
from src.exception import CustomException
from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)

# ── Flask App ──────────────────────────────────
app = Flask(__name__)
app.secret_key = 'super_secret_medico_key'
# Database Configuration (MongoDB)
app.config['UPLOAD_FOLDER'] = os.path.join('artifacts', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['medico']

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = user_data.get('name')
        self.email = user_data.get('email')
        self.password = user_data.get('password')

# Seed health tips if empty or fewer than 10
if db.health_tips.count_documents({}) < 10:
    db.health_tips.delete_many({}) # Clear existing to avoid duplicates when adding the new set
    tips = [
        {"title": "Stay Hydrated", "content": "Drinking enough water every day is crucial for many reasons: to regulate body temperature, keep joints lubricated, prevent infections, deliver nutrients to cells, and keep organs functioning properly. Being well-hydrated also improves sleep quality, cognition, and mood.", "date_added": datetime.utcnow()},
        {"title": "Benefits of Walking", "content": "Walking for 30 minutes a day or more on most days of the week is a great way to improve or maintain your overall health. If you can't manage 30 minutes a day, remember even short walks are better than none at all.", "date_added": datetime.utcnow()},
        {"title": "Healthy Sleep Habits", "content": "A healthy adult needs between 7 and 9 hours of sleep per night. Going to bed and waking up at the same time every day can help improve your sleep quality.", "date_added": datetime.utcnow()},
        {"title": "Eat More Fiber", "content": "Dietary fiber, found mainly in fruits, vegetables, whole grains and legumes, is probably best known for its ability to prevent or relieve constipation. But foods containing fiber can provide other health benefits as well, such as helping to maintain a healthy weight and lowering your risk of diabetes, heart disease and some types of cancer.", "date_added": datetime.utcnow()},
        {"title": "Mindful Eating", "content": "Mindful eating is about using mindfulness to reach a state of full attention to your experiences, cravings, and physical cues when eating. This practice helps you learn to hear what your body is telling you about hunger and satisfaction.", "date_added": datetime.utcnow()},
        {"title": "Limit Added Sugars", "content": "Consuming too much added sugar is linked to an increased risk of weight gain, obesity, type 2 diabetes, and heart disease. Check food labels and try to choose products with little to no added sugar.", "date_added": datetime.utcnow()},
        {"title": "Regular Screen Breaks", "content": "To prevent eye strain from digital devices, follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for at least 20 seconds. Also, remember to blink frequently to keep your eyes moist.", "date_added": datetime.utcnow()},
        {"title": "Strength Training", "content": "Incorporating strength training exercises at least two days a week is essential. It helps increase muscle mass, strengthens your bones, and boosts your metabolism, which can help with weight management.", "date_added": datetime.utcnow()},
        {"title": "Practice Deep Breathing", "content": "Taking a few minutes each day to practice deep breathing can significantly reduce stress levels. It signals your nervous system to calm down, lowers your heart rate, and can help reduce blood pressure.", "date_added": datetime.utcnow()},
        {"title": "Don't Skip Breakfast", "content": "Eating a nutritious breakfast kick-starts your metabolism and provides you with the energy you need to focus and be productive throughout the morning. Opt for protein and whole grains over sugary pastries.", "date_added": datetime.utcnow()}
    ]
    db.health_tips.insert_many(tips)

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = db.users.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except Exception:
        pass
    return None

@app.context_processor
def inject_emergency_contacts():
    if current_user.is_authenticated:
        contacts = list(db.emergency_contacts.find({'user_id': current_user.id}))
        for c in contacts:
            c['id'] = str(c['_id'])
        return dict(global_emergency_contacts=contacts)
    return dict(global_emergency_contacts=[])

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

        # Save to SymptomHistory
        new_history = {
            'user_id': current_user.id,
            'symptoms': request.form.get("symptoms"),
            'severity': request.form.get("severity"),
            'predicted_medicine': result["medicine"],
            'date_added': datetime.utcnow()
        }
        db.symptom_history.insert_one(new_history)

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
        
        user = db.users.find_one({'email': email})
        if user:
            flash('Email address already exists')
            return redirect(url_for('register'))
            
        new_user_data = {
            'name': name, 
            'email': email, 
            'password': generate_password_hash(password, method='pbkdf2:sha256')
        }
        res = db.users.insert_one(new_user_data)
        new_user_data['_id'] = res.inserted_id
        
        login_user(User(new_user_data))
        return redirect(url_for('predict'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = db.users.find_one({'email': email})
        
        if not user_data or not check_password_hash(user_data['password'], password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
            
        login_user(User(user_data))
        return redirect(url_for('predict'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    name = request.form.get('name')
    email = request.form.get('email')
    
    if name and email:
        existing = db.users.find_one({'email': email})
        if existing and str(existing['_id']) != current_user.id:
            flash("Email already in use.")
        else:
            db.users.update_one(
                {'_id': ObjectId(current_user.id)}, 
                {'$set': {'name': name, 'email': email}}
            )
            current_user.name = name
            current_user.email = email
            flash("Profile updated successfully.")
            
    return redirect(url_for('dashboard'))

# ── ADMIN PANEL ─────────────────────────────────
@app.route('/admin')
@login_required
def admin_panel():
    if current_user.email != 'admin@medico.com':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('dashboard'))
    
    total_users = db.users.count_documents({})
    total_records = db.medical_records.count_documents({})
    total_checks = db.symptom_history.count_documents({})
    recent_users = list(db.users.find().sort('_id', -1).limit(5))
    for u in recent_users: u['id'] = str(u['_id'])
    recent_checks = list(db.symptom_history.find().sort('date_added', -1).limit(10))
    for c in recent_checks: c['id'] = str(c['_id'])
    
    return render_template('admin.html', 
                           total_users=total_users, 
                           total_records=total_records, 
                           total_checks=total_checks,
                           recent_users=recent_users,
                           recent_checks=recent_checks)

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
            
        new_record = {
            'user_id': current_user.id,
            'record_type': record_type,
            'description': description,
            'file_path': file_path,
            'date_added': datetime.utcnow()
        }
        db.medical_records.insert_one(new_record)
        flash('Record added successfully')
        return redirect(url_for('dashboard'))
        
    records = list(db.medical_records.find({'user_id': current_user.id}).sort('date_added', -1))
    for r in records: r['id'] = str(r['_id'])
    symptom_history = list(db.symptom_history.find({'user_id': current_user.id}).sort('date_added', -1))
    for s in symptom_history: s['id'] = str(s['_id'])
    contacts = list(db.emergency_contacts.find({'user_id': current_user.id}))
    for c in contacts: c['id'] = str(c['_id'])
    return render_template('dashboard.html', user=current_user, records=records, symptom_history=symptom_history, contacts=contacts)

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view_record/<record_id>')
@login_required
def view_record(record_id):
    record = db.medical_records.find_one({'_id': ObjectId(record_id)})
    if not record:
        flash('Record not found')
        return redirect(url_for('dashboard'))
    if record.get('user_id') != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    if not record.get('file_path'):
        flash('No file associated with this record')
        return redirect(url_for('dashboard'))
        
    filename = os.path.basename(record['file_path'])
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/edit_record/<record_id>', methods=['POST'])
@login_required
def edit_record(record_id):
    record = db.medical_records.find_one({'_id': ObjectId(record_id)})
    if not record:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    if record.get('user_id') != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    update_data = {}
    record_type = request.form.get('record_type')
    if record_type: update_data['record_type'] = record_type
    description = request.form.get('description')
    if description: update_data['description'] = description
    
    file = request.files.get('file')
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        update_data['file_path'] = file_path
        
    if update_data:
        db.medical_records.update_one({'_id': ObjectId(record_id)}, {'$set': update_data})
    
    flash('Record updated successfully')
    return redirect(url_for('dashboard'))

@app.route('/delete_record/<record_id>')
@login_required
def delete_record(record_id):
    record = db.medical_records.find_one({'_id': ObjectId(record_id)})
    if not record:
        flash('Record not found')
        return redirect(url_for('dashboard'))
    if record.get('user_id') != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    db.medical_records.delete_one({'_id': ObjectId(record_id)})
    flash('Record deleted')
    return redirect(url_for('dashboard'))

# ── EMERGENCY CONTACTS ───────────────────────
@app.route('/add_contact', methods=['POST'])
@login_required
def add_contact():
    name = request.form.get('name')
    phone = request.form.get('phone')
    new_contact = {'user_id': current_user.id, 'name': name, 'phone': phone}
    db.emergency_contacts.insert_one(new_contact)
    flash('Contact added')
    return redirect(url_for('dashboard'))

@app.route('/delete_contact/<contact_id>')
@login_required
def delete_contact(contact_id):
    contact = db.emergency_contacts.find_one({'_id': ObjectId(contact_id)})
    if contact and contact.get('user_id') == current_user.id:
        db.emergency_contacts.delete_one({'_id': ObjectId(contact_id)})
        flash('Contact deleted')
    return redirect(url_for('dashboard'))

# ── MEDICINE INTERACTION CHECKER ────────────────
@app.route("/interactions", methods=["GET", "POST"])
def interactions():
    if request.method == "GET":
        return render_template("interactions.html")
    
    med1 = request.form.get("medicine_1", "").lower()
    med2 = request.form.get("medicine_2", "").lower()
    
    # Mocking interaction logic
    interactions_db = {
        ("aspirin", "ibuprofen"): "May increase risk of bleeding and stomach ulcers.",
        ("paracetamol", "alcohol"): "May increase risk of liver damage.",
        ("lisinopril", "potassium"): "May cause hyperkalemia (high potassium levels).",
    }
    
    interaction_result = "No known major interactions between these medications based on our limited database. However, always consult your doctor."
    for (m1, m2), msg in interactions_db.items():
        if (m1 in med1 and m2 in med2) or (m1 in med2 and m2 in med1):
            interaction_result = f"Warning: {msg}"
            break
            
    return render_template("interactions.html", med1=med1, med2=med2, result=interaction_result)

# ── MEDICATION REMINDERS ────────────────
@app.route("/reminders", methods=["GET", "POST"])
@login_required
def reminders():
    if request.method == "POST":
        medicine_name = request.form.get("medicine_name")
        dosage = request.form.get("dosage")
        time = request.form.get("time")
        
        new_reminder = {
            'user_id': current_user.id,
            'medicine_name': medicine_name,
            'dosage': dosage,
            'time': time,
            'date_added': datetime.utcnow()
        }
        db.medication_reminders.insert_one(new_reminder)
        flash("Reminder added successfully!")
        return redirect(url_for("reminders"))
        
    user_reminders = list(db.medication_reminders.find({'user_id': current_user.id}))
    for r in user_reminders: r['id'] = str(r['_id'])
    return render_template("reminders.html", reminders=user_reminders)

@app.route('/delete_reminder/<reminder_id>')
@login_required
def delete_reminder(reminder_id):
    reminder = db.medication_reminders.find_one({'_id': ObjectId(reminder_id)})
    if reminder and reminder.get('user_id') == current_user.id:
        db.medication_reminders.delete_one({'_id': ObjectId(reminder_id)})
        flash('Reminder deleted')
    return redirect(url_for('reminders'))

# ── HEALTH TIPS ────────────────
@app.route("/health_tips")
def health_tips():
    tips = list(db.health_tips.find().sort('date_added', -1))
    for t in tips: t['id'] = str(t['_id'])
    return render_template("health_tips.html", tips=tips)

@app.route('/api/reminders')
@login_required
def api_reminders():
    user_reminders = list(db.medication_reminders.find({'user_id': current_user.id}))
    reminders_data = [{"medicine_name": r.get('medicine_name'), "time": r.get('time')} for r in user_reminders]
    return jsonify({"reminders": reminders_data})

# ── BMI CALCULATOR ──────────────────────────────
@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    bmi_result = None
    category = None
    color = "success"
    if request.method == 'POST':
        try:
            weight = float(request.form.get('weight'))
            height_cm = float(request.form.get('height'))
            height_m = height_cm / 100
            bmi_value = weight / (height_m * height_m)
            bmi_result = round(bmi_value, 1)
            
            if bmi_result < 18.5:
                category = "Underweight"
                color = "warning"
            elif 18.5 <= bmi_result < 25:
                category = "Normal weight"
                color = "success"
            elif 25 <= bmi_result < 30:
                category = "Overweight"
                color = "warning"
            else:
                category = "Obese"
                color = "alert"
        except (TypeError, ValueError):
            flash("Please enter valid numbers.")
            
    return render_template('bmi.html', bmi=bmi_result, category=category, color=color)



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
    port=int(os.environ.get("PORT",5000))
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = True
    )
