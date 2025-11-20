import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for

# --- Model Definition (Paste your class here) ---
class CrackCNN(nn.Module):
    def __init__(self):
        super(CrackCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        # --- dynamically calculate flattened size ---
        dummy_input = torch.zeros(1, 3, 128, 128)  # batch size 1
        dummy_output = self._forward_features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        # ------------------------------------------

        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)  # flatten all except batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrackCNN().to(device)
model_path = 'best_model.pth'

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'best_model.pth' is in the same directory and contains the state_dict.")
else:
    print(f"Warning: '{model_path}' not found. Predictions will be random.")

model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess
        try:
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Assuming class 0 is Negative (Not Cracked) and class 1 is Positive (Cracked)
                label = "Cracked" if predicted_class.item() == 1 else "Not Cracked"
                conf_score = confidence.item() * 100

            # Generate correct URL for the image
            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return render_template('index.html', 
                                   prediction=label, 
                                   confidence=f"{conf_score:.2f}%", 
                                   image_url=image_url)
        except Exception as e:
            return f"Error processing image: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
