import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ============================================================================
# Configuration
# ============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model parameters (must match training)
MAX_QUESTION_LENGTH = 25

# ============================================================================
# Load Model and Resources
# ============================================================================
print("Loading VQA model and resources...")

# Load trained VQA model
vqa_model = load_model('vqa_model_50k.h5')
print("✅ VQA model loaded")

# Load question tokenizer
with open('question_tokenizer.pkl', 'rb') as f:
    question_tokenizer = pickle.load(f)
print("✅ Question tokenizer loaded")

# Load answer mappings
with open('answer_mappings.pkl', 'rb') as f:
    data = pickle.load(f)
    answer_to_idx = data['answer_to_idx']
    idx_to_answer = data['idx_to_answer']
print("✅ Answer mappings loaded")

# Create feature extractor
print("Creating feature extractor...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)
print("✅ Feature extractor ready")

print("\n🚀 All resources loaded successfully!")

# ============================================================================
# Helper Functions
# ============================================================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image for feature extraction"""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features

def process_question(question):
    """Process question for model input"""
    q_seq = question_tokenizer.texts_to_sequences([question])
    q_pad = pad_sequences(q_seq, maxlen=MAX_QUESTION_LENGTH, padding='post')
    return q_pad

def predict_answer(image_path, question, top_k=5):
    """Get predictions for image + question"""
    # Process inputs
    img_features = process_image(image_path)
    q_processed = process_question(question)
    
    # Get predictions
    predictions = vqa_model.predict([img_features, q_processed], verbose=0)[0]
    
    # Get top k answers
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        answer = idx_to_answer.get(idx, "unknown")
        confidence = float(predictions[idx]) * 100
        results.append({
            'answer': answer,
            'confidence': round(confidence, 2)
        })
    
    return results

# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    question = request.form.get('question', '').strip()
    
    # Validate inputs
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictions
        predictions = predict_answer(filepath, question, top_k=5)
        
        return jsonify({
            'success': True,
            'question': question,
            'predictions': predictions,
            'image_url': f'/static/uploads/{filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

# ============================================================================
# Run Application
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("VQA Flask Application")
    print("="*60)
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)