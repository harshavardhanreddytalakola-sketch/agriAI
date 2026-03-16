PlantCareAI – Plant Disease Detection System

Author: Harshavardhanreddy Talakola

Project Description
PlantCareAI is a deep learning based web application that detects plant diseases from leaf images. The system uses a trained MobileNetV2 convolutional neural network model to classify plant leaf images into 38 different disease categories or healthy conditions. The application allows users to upload an image of a plant leaf and receive predictions along with disease information, symptoms, treatment suggestions, and prevention methods.

Technology Stack
Backend: Python, Flask
Machine Learning: TensorFlow, Keras
Model Architecture: MobileNetV2
Image Processing: PIL (Pillow), NumPy
Frontend: HTML (Flask Templates)
Other Libraries: base64, io, os

Features
• Upload plant leaf images (PNG, JPG, JPEG, WEBP)
• Detect 38 plant diseases and healthy conditions
• Display prediction confidence score
• Show top 3 predicted diseases
• Provide disease description
• Show symptoms, treatment, and prevention steps
• Image preview of uploaded leaf

Dataset
The model is trained on the PlantVillage dataset containing multiple plant species and diseases. The system supports 38 classes including crops such as:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

Project Structure
project/
│
├── app.py
├── mobilenetv2_best.keras
├── templates/
│ ├── home.html
│ └── about.html
├── static/
└── README.txt

How the System Works
1. User uploads a plant leaf image.
2. The image is resized to 224x224 pixels.
3. MobileNetV2 preprocessing is applied.
4. The trained model predicts the disease class.
5. The system returns:
   - Predicted disease
   - Confidence score
   - Top 3 predictions
   - Disease information including symptoms and treatment.

Installation
1. Clone the repository
   git clone https://github.com/yourusername/plantcareai.git

2. Navigate to the project directory
   cd plantcareai

3. Install required dependencies
   pip install flask tensorflow numpy pillow

4. Run the application
   python app.py

5. Open in browser
   http://127.0.0.1:5000

API Endpoint
POST /predict

Input:
Image file

Output:
JSON containing:
- predicted disease
- confidence score
- top 3 predictions
- disease details

Example Use Case
Farmers or gardeners can upload a photo of a plant leaf to instantly identify diseases and receive treatment and prevention recommendations.

Future Improvements
• Mobile application integration
• Real-time camera detection
• Larger dataset for improved accuracy
• Support for more crop species
• Deployment on cloud platforms

License
This project is created for educational and research purposes.
