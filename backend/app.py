import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model.predict import PlantDiseasePredictor
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

predictor = PlantDiseasePredictor()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = predictor.predict(filepath)
        os.remove(filepath)

        if results is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first by running: python model/train_model.py'
            }), 503

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'classes': len(Config.CLASSES)
    })

if __name__ == '__main__':
    print("\n🌿 PlantCare AI Flask Server")
    print("=" * 40)
    print(f"   Model loaded: {predictor.model is not None}")
    print(f"   Disease classes: {len(Config.CLASSES)}")
    print(f"   Visit: http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, port=5000)

