from flask import Flask, request,session, render_template, redirect, url_for,jsonify
from werkzeug.security import generate_password_hash,check_password_hash
from flask_pymongo import PyMongo
from flask_session import Session

from functools import wraps

# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from torchvision import models
import os



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'dc25735aee9b7509ceff473e929b08e49f6bdf85'
app.config["MONGO_URI"]="mongodb+srv://faizanazam6980:gX6Fv5ckklb6AqWk@cluster0.tjketqg.mongodb.net/FypDatabase?retryWrites=true&w=majority&appName=Cluster0"


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

Session(app)

mongodb_client=PyMongo(app)
db=mongodb_client.db

# Load your model
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# model_path = './saved_models/resnet152-b121ed2d.pth'
# model = models.resnet152(weights=None)
# model.load_state_dict(torch.load(model_path))
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 5)  # Change the final layer to have 5 output features
# model.eval()

# labels = ['cocci', 'healthy', 'new_castle', 'salmo', 'white_diarrhea']

# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = preprocess(image)
#     image = image.unsqueeze(0)
#     return image

# def predict(image_path):
#     image = load_image(image_path)
    
#     with torch.no_grad():
#         outputs = model(image)
    
#     _, predicted_class = torch.max(outputs, 1)
    
#     predicted_class = predicted_class.item()  # Convert tensor to integer
#     return labels[predicted_class]


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/home')
@login_required
def index():
    return render_template('index.html')

@app.route('/about')
@login_required

def about():
    return render_template('about.html')

@app.route('/research')
@login_required

def research():
    return render_template('Research.html')

@app.route('/contact')
@login_required

def contact():
    return render_template('contact.html')
    


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict(file_path)
            # Redirect to result with prediction and relative file path
            return redirect(url_for('result', prediction=prediction, file_name=file.filename))
    return 'Failed to upload file'

# Route to show result
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    file_name = request.args.get('file_name')
    file_path = os.path.join('uploads', file_name)  # Relative path from 'static'
    return render_template('result.html', prediction=prediction, file_path=file_path)


# Signup The USer
@app.route('/signupuser', methods=['POST'])
def signupuser():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    cnic = data.get('cnic')
    password = data.get('password')

    if name and email and phone and cnic and password:
        # Check if the email, phone, or CNIC already exists
        if db.users.find_one({"email": email}):
            return jsonify({"error": "Email already exists!"}), 400
        if db.users.find_one({"phone": phone}):
            return jsonify({"error": "Phone number already exists!"}), 400
        if db.users.find_one({"cnic": cnic}):
            return jsonify({"error": "CNIC already exists!"}), 400

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        # Create a user document
        user = {
            "name": name,
            "email": email,
            "phone": phone,
            "cnic": cnic,
            "password": hashed_password
        }

        # Insert the user into the database
        db.users.insert_one(user)

        return jsonify({"message": "User registered successfully!"}), 201
    else:
        return jsonify({"error": "Missing fields!"}), 400

# Login the user
@app.route('/loginuser', methods=['POST'])
def loginuser():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if email and password:
        user = db.users.find_one({"email": email})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['name']
            return jsonify({"message": "Login successful!"}), 200
        else:
            return jsonify({"error": "Invalid email or password!"}), 401
    else:
        return jsonify({"error": "Missing fields!"}), 400

# Logout the user
@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('login'))



# Signup The USer
@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    email = data.get('email')

    if email:
        # Check if the email already exists
        if db.subscribe.find_one({"email": email}):
            return jsonify({"error": "Email already exists!"}), 400

        # Create a subscription document
        user = {
            "email": email,
        }

        # Insert the subscription into the database
        db.subscribe.insert_one(user)

        return jsonify({"message": "User subscribed successfully!"}), 201
    else:
        return jsonify({"error": "Missing fields!"}), 400


@app.route('/connection')
def connection():
    return "MongoDB connection established successfully."

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
