import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilenetv2_best.keras")
model = None

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

# ── 38 class labels (New Plant Diseases Dataset — alphabetical order) ────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── Disease info database ────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display": "Apple Scab",
        "plant": "Apple",
        "status": "diseased",
        "severity": "Moderate",
        "description": "A fungal disease caused by Venturia inaequalis that causes dark, scabby lesions on leaves and fruit.",
        "symptoms": ["Dark olive-green spots on leaves", "Scabby, cracked lesions on fruit", "Premature leaf drop", "Deformed fruit"],
        "treatment": ["Apply fungicides (captan, myclobutanil) at bud break", "Remove and destroy infected leaves and fruit", "Prune to improve air circulation", "Plant scab-resistant varieties"],
        "prevention": ["Rake and dispose of fallen leaves", "Apply dormant sprays in early spring", "Avoid overhead irrigation", "Space trees for good air flow"],
    },
    "Apple___Black_rot": {
        "display": "Apple Black Rot",
        "plant": "Apple",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Botryosphaeria obtusa, this fungal disease affects fruit, leaves, and bark causing significant losses.",
        "symptoms": ["Brown leaf spots with purple margins", "Rotting fruit with concentric rings", "Cankers on branches", "Mummified fruit"],
        "treatment": ["Prune out infected wood 10–15 cm below visible symptoms", "Apply copper-based fungicides", "Remove mummified fruit", "Improve canopy airflow"],
        "prevention": ["Remove dead wood regularly", "Avoid bark injuries", "Apply protective fungicide sprays", "Maintain tree vigor with proper fertilization"],
    },
    "Apple___Cedar_apple_rust": {
        "display": "Cedar Apple Rust",
        "plant": "Apple",
        "status": "diseased",
        "severity": "Moderate",
        "description": "A fungal disease requiring two hosts (apple and cedar/juniper) to complete its life cycle.",
        "symptoms": ["Bright orange-yellow spots on upper leaf surface", "Tube-like structures on leaf undersides", "Deformed, discolored fruit", "Premature defoliation"],
        "treatment": ["Apply fungicides (myclobutanil, trifloxystrobin) from pink bud stage", "Remove nearby juniper/cedar trees if possible", "Apply preventive sprays every 7–10 days during wet weather"],
        "prevention": ["Plant rust-resistant apple varieties", "Avoid planting near eastern red cedars", "Monitor and treat cedars for galls in spring"],
    },
    "Apple___healthy": {
        "display": "Healthy Apple",
        "plant": "Apple",
        "status": "healthy",
        "severity": "None",
        "description": "Your apple plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Maintain regular watering schedule", "Apply balanced fertilizer annually", "Monitor for early signs of pests or disease", "Prune annually for good air circulation"],
    },
    "Blueberry___healthy": {
        "display": "Healthy Blueberry",
        "plant": "Blueberry",
        "status": "healthy",
        "severity": "None",
        "description": "Your blueberry plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Maintain soil pH between 4.5–5.5", "Mulch with pine bark or sawdust", "Water consistently", "Prune old canes for renewal"],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry Powdery Mildew",
        "plant": "Cherry",
        "status": "diseased",
        "severity": "Moderate",
        "description": "A fungal disease caused by Podosphaera clandestina that creates white powdery growth on leaves and shoots.",
        "symptoms": ["White powdery patches on leaves", "Distorted young shoots", "Infected leaves curl upward", "Premature defoliation in severe cases"],
        "treatment": ["Apply sulfur or potassium bicarbonate sprays", "Use systemic fungicides (myclobutanil, trifloxystrobin)", "Remove heavily infected shoots", "Avoid excessive nitrogen fertilization"],
        "prevention": ["Plant resistant varieties", "Ensure good air circulation through pruning", "Avoid overhead watering", "Apply preventive fungicide in spring"],
    },
    "Cherry_(including_sour)___healthy": {
        "display": "Healthy Cherry",
        "plant": "Cherry",
        "status": "healthy",
        "severity": "None",
        "description": "Your cherry plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Prune annually after harvest", "Monitor for brown rot during wet weather", "Avoid wounding the bark", "Apply balanced fertilizer each spring"],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Corn Gray Leaf Spot",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Cercospora zeae-maydis, this is one of the most yield-limiting diseases of corn worldwide.",
        "symptoms": ["Rectangular tan to gray lesions parallel to leaf veins", "Lesions with yellow halos", "Lesions merge causing blighting of entire leaves", "Premature plant death in severe cases"],
        "treatment": ["Apply fungicides (strobilurin, triazole) at early disease onset", "Reduce plant density for better airflow", "Avoid overhead irrigation"],
        "prevention": ["Plant resistant hybrids", "Rotate crops with non-host plants", "Till crop residue to reduce inoculum", "Avoid continuous corn monoculture"],
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn Common Rust",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Puccinia sorghi, common rust produces brick-red pustules on corn leaves.",
        "symptoms": ["Small, circular to elongated brick-red pustules on both leaf surfaces", "Pustules turn black late in season", "Severe infections cause premature leaf death"],
        "treatment": ["Apply fungicides (triazoles, strobilurins) if infection is early and severe", "Scout fields regularly during tasseling"],
        "prevention": ["Plant resistant corn hybrids", "Early planting to avoid peak spore periods", "Crop rotation"],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn Northern Leaf Blight",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Exserohilum turcicum, this disease causes large cigar-shaped lesions on corn leaves.",
        "symptoms": ["Long, elliptical grayish-green or tan lesions (2.5–15 cm)", "Lesions may have dark green 'water-soaked' border", "Severe blighting and early death of leaves"],
        "treatment": ["Fungicide application (strobilurin, triazole mixtures) before disease reaches upper canopy", "Remove crop debris after harvest"],
        "prevention": ["Plant resistant hybrids", "Crop rotation", "Tillage to reduce residue-borne inoculum"],
    },
    "Corn_(maize)___healthy": {
        "display": "Healthy Corn",
        "plant": "Corn (Maize)",
        "status": "healthy",
        "severity": "None",
        "description": "Your corn plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Rotate crops annually", "Test soil and apply appropriate nutrients", "Scout fields regularly", "Use certified disease-free seed"],
    },
    "Grape___Black_rot": {
        "display": "Grape Black Rot",
        "plant": "Grape",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Guignardia bidwellii, black rot can destroy entire grape crops in warm, wet conditions.",
        "symptoms": ["Brown circular lesions on leaves with dark borders", "Mummified, shriveled black berries", "Black pycnidia (fruiting bodies) visible in lesions"],
        "treatment": ["Apply fungicides (mancozeb, myclobutanil) from bud break through veraison", "Remove all mummified berries and infected plant material"],
        "prevention": ["Ensure good canopy management and airflow", "Remove overwintering inoculum", "Avoid working in vineyard when wet"],
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Grape Esca (Black Measles)",
        "plant": "Grape",
        "status": "diseased",
        "severity": "High",
        "description": "A complex wood disease caused by multiple fungal pathogens causing internal wood decay.",
        "symptoms": ["Interveinal chlorosis and necrosis (tiger-stripe pattern)", "Small, dark spots on berries", "Shoot dieback and sudden vine collapse"],
        "treatment": ["No cure; manage symptom expression", "Remove and destroy infected wood", "Protect pruning wounds with fungicide paste"],
        "prevention": ["Prune during dry weather", "Apply wound sealants after pruning", "Avoid large pruning wounds", "Replace severely affected vines"],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape Leaf Blight",
        "plant": "Grape",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Pseudocercospora vitis, this disease causes irregular leaf spots and defoliation.",
        "symptoms": ["Irregular dark brown spots on leaves", "Premature defoliation", "Weakened vines with reduced fruit quality"],
        "treatment": ["Apply copper-based or mancozeb fungicides", "Remove fallen infected leaves"],
        "prevention": ["Maintain good canopy airflow", "Avoid overhead irrigation", "Mulch to prevent soil splash"],
    },
    "Grape___healthy": {
        "display": "Healthy Grape",
        "plant": "Grape",
        "status": "healthy",
        "severity": "None",
        "description": "Your grape plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Regular pruning for airflow", "Monitor for powdery and downy mildew", "Maintain balanced soil nutrition", "Irrigate at base to keep foliage dry"],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Citrus Greening (HLB)",
        "plant": "Orange",
        "status": "diseased",
        "severity": "Critical",
        "description": "Huanglongbing (HLB) is the most devastating citrus disease worldwide, caused by Candidatus Liberibacter bacteria transmitted by Asian citrus psyllid.",
        "symptoms": ["Asymmetric yellowing of leaves (blotchy mottle)", "Small, lopsided fruit with bitter taste", "Dieback of shoots", "Green fruit that doesn't color properly"],
        "treatment": ["No cure exists; infected trees eventually die", "Manage Asian citrus psyllid with insecticides", "Remove and destroy infected trees to prevent spread", "Nutritional supplements may slow decline"],
        "prevention": ["Plant certified disease-free nursery stock", "Control psyllid populations", "Quarantine measures when introducing new plants", "Monitor regularly for psyllid and symptoms"],
    },
    "Peach___Bacterial_spot": {
        "display": "Peach Bacterial Spot",
        "plant": "Peach",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Xanthomonas arboricola pv. pruni, bacterial spot affects leaves, fruit, and twigs.",
        "symptoms": ["Small water-soaked spots on leaves turning purple-brown", "Premature defoliation", "Sunken, corky lesions on fruit", "Dark cankers on twigs"],
        "treatment": ["Apply copper bactericides in spring", "Use oxytetracycline sprays during bloom if permitted", "Avoid excessive nitrogen fertilization"],
        "prevention": ["Plant resistant varieties", "Avoid working in orchard when wet", "Improve air circulation through pruning", "Site selection with good drainage"],
    },
    "Peach___healthy": {
        "display": "Healthy Peach",
        "plant": "Peach",
        "status": "healthy",
        "severity": "None",
        "description": "Your peach plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Annual thinning for good fruit size", "Prune for open-vase shape", "Monitor for peach leaf curl in spring", "Apply dormant oil spray in late winter"],
    },
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper Bacterial Spot",
        "plant": "Bell Pepper",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Xanthomonas campestris pv. vesicatoria, this disease thrives in warm, wet conditions.",
        "symptoms": ["Small water-soaked lesions on leaves turning brown with yellow halos", "Raised, scabby spots on fruit", "Premature defoliation", "Reduced yield"],
        "treatment": ["Apply copper bactericides + mancozeb", "Avoid overhead irrigation", "Remove and destroy infected plant material"],
        "prevention": ["Use disease-free certified seed", "Practice crop rotation (3-year)", "Plant resistant varieties", "Disinfect tools and equipment"],
    },
    "Pepper,_bell___healthy": {
        "display": "Healthy Bell Pepper",
        "plant": "Bell Pepper",
        "status": "healthy",
        "severity": "None",
        "description": "Your bell pepper plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Water at the base, not overhead", "Rotate crops each season", "Mulch to reduce soil splash", "Scout regularly for aphids and thrips"],
    },
    "Potato___Early_blight": {
        "display": "Potato Early Blight",
        "plant": "Potato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Alternaria solani, early blight is common in stressed or aging potato plants.",
        "symptoms": ["Dark brown circular spots with concentric rings (target-like)", "Yellow halo around spots", "Lower leaves affected first", "Defoliation in severe cases"],
        "treatment": ["Apply fungicides (chlorothalonil, mancozeb, azoxystrobin)", "Remove heavily infected leaves", "Ensure adequate plant nutrition"],
        "prevention": ["Use certified disease-free seed tubers", "Avoid overhead irrigation", "Maintain proper plant spacing", "Crop rotation with non-solanaceous crops"],
    },
    "Potato___Late_blight": {
        "display": "Potato Late Blight",
        "plant": "Potato",
        "status": "diseased",
        "severity": "Critical",
        "description": "Caused by Phytophthora infestans — the same pathogen behind the Irish Famine — late blight can devastate crops rapidly.",
        "symptoms": ["Water-soaked pale green lesions that turn brown-black", "White fluffy sporulation on leaf undersides in humid conditions", "Rapid collapse of foliage", "Tuber rot with reddish-brown discoloration"],
        "treatment": ["Apply fungicides immediately (metalaxyl, chlorothalonil, cymoxanil)", "Destroy infected plant material", "Hill up tubers to protect from spores", "Harvest early if infection is severe"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Scout weekly during cool, wet weather", "Use certified disease-free seed"],
    },
    "Potato___healthy": {
        "display": "Healthy Potato",
        "plant": "Potato",
        "status": "healthy",
        "severity": "None",
        "description": "Your potato plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Use certified seed potatoes", "Hill plants properly", "Monitor weekly for blight signs", "Practice crop rotation every 3+ years"],
    },
    "Raspberry___healthy": {
        "display": "Healthy Raspberry",
        "plant": "Raspberry",
        "status": "healthy",
        "severity": "None",
        "description": "Your raspberry plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Prune out old floricanes after harvest", "Ensure good drainage", "Space plants 60 cm apart for airflow", "Monitor for cane blight and spur blight"],
    },
    "Soybean___healthy": {
        "display": "Healthy Soybean",
        "plant": "Soybean",
        "status": "healthy",
        "severity": "None",
        "description": "Your soybean plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Use certified disease-free seed", "Rotate with non-legume crops", "Scout for sudden death syndrome and SCN", "Maintain proper soil pH 6.0–6.8"],
    },
    "Squash___Powdery_mildew": {
        "display": "Squash Powdery Mildew",
        "plant": "Squash",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Podosphaera xanthii and Erysiphe cichoracearum, powdery mildew is very common in squash.",
        "symptoms": ["White powdery coating on leaf surfaces", "Yellowing and browning of infected leaves", "Premature leaf death", "Reduced fruit yield and quality"],
        "treatment": ["Apply potassium bicarbonate, sulfur, or neem oil sprays", "Use systemic fungicides (myclobutanil) for severe infections", "Remove heavily infected leaves"],
        "prevention": ["Plant resistant varieties", "Avoid excessive nitrogen", "Ensure good spacing and airflow", "Water early in the day"],
    },
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry Leaf Scorch",
        "plant": "Strawberry",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Diplocarpon earlianum, leaf scorch produces dark purple spots that can reduce plant vigor.",
        "symptoms": ["Small, irregular dark purple spots on upper leaf surface", "Spots enlarge and coalesce giving a 'scorched' look", "Leaf margins turn brown and die", "Reduced runner production"],
        "treatment": ["Apply fungicides (captan, myclobutanil) after first symptom appearance", "Remove and destroy infected leaves"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Mulch to prevent soil splash", "Renovate beds by mowing and thinning after harvest"],
    },
    "Strawberry___healthy": {
        "display": "Healthy Strawberry",
        "plant": "Strawberry",
        "status": "healthy",
        "severity": "None",
        "description": "Your strawberry plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Renovate beds annually", "Use certified plants", "Avoid planting in soil with Verticillium history", "Water at the base"],
    },
    "Tomato___Bacterial_spot": {
        "display": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "High",
        "description": "Caused by Xanthomonas species, bacterial spot thrives in warm, wet weather and can cause severe losses.",
        "symptoms": ["Small, water-soaked spots on leaves with yellow halos", "Raised, scabby spots on fruit", "Defoliation of lower leaves", "Reduced fruit quality"],
        "treatment": ["Apply copper bactericides + mancozeb sprays", "Avoid working in wet fields", "Remove infected plant material"],
        "prevention": ["Use disease-free certified seed", "Hot-water treat seed before planting", "Crop rotation (2+ years)", "Avoid overhead irrigation"],
    },
    "Tomato___Early_blight": {
        "display": "Tomato Early Blight",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Alternaria solani, early blight is widespread and affects foliage, stems, and fruit.",
        "symptoms": ["Dark brown spots with concentric target rings on older leaves", "Yellow halo around lesions", "Collar rot on stems of seedlings", "Dark, leathery lesions on fruit near stem"],
        "treatment": ["Apply fungicides (chlorothalonil, mancozeb, azoxystrobin)", "Remove affected leaves promptly", "Stake plants to improve airflow"],
        "prevention": ["Mulch to prevent soil splash", "Avoid overhead watering", "Crop rotation", "Ensure good plant nutrition"],
    },
    "Tomato___Late_blight": {
        "display": "Tomato Late Blight",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Critical",
        "description": "Caused by Phytophthora infestans, late blight can destroy entire tomato fields within days under favorable conditions.",
        "symptoms": ["Water-soaked olive-green to brown lesions on leaves", "White sporulation on leaf undersides in humid conditions", "Dark brown greasy spots on fruit", "Rapid plant collapse"],
        "treatment": ["Apply fungicides immediately (metalaxyl, chlorothalonil, cymoxanil)", "Remove and bag infected plants immediately", "Do not compost infected material"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Scout after cool, wet weather", "Destroy volunteer tomato and potato plants"],
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato Leaf Mold",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Passalora fulva (formerly Cladosporium fulvum), leaf mold is common in greenhouse tomatoes.",
        "symptoms": ["Pale green to yellow spots on upper leaf surface", "Olive-green to brown velvety mold on leaf undersides", "Leaves curl and die", "Fruit infection rare but possible"],
        "treatment": ["Apply fungicides (mancozeb, chlorothalonil, copper-based)", "Improve greenhouse ventilation", "Reduce humidity below 85%"],
        "prevention": ["Plant resistant varieties", "Ensure good airflow with proper spacing", "Avoid wetting foliage", "Sanitize greenhouse between crops"],
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Septoria lycopersici, this is one of the most destructive tomato foliage diseases.",
        "symptoms": ["Numerous small circular spots with dark brown margins and light grey centres", "Tiny black pycnidia visible in lesion centres", "Premature defoliation starting from lower leaves", "Reduced fruit size and quality"],
        "treatment": ["Apply fungicides (chlorothalonil, mancozeb, copper-based)", "Remove infected lower leaves", "Stake and mulch plants"],
        "prevention": ["Crop rotation (2+ years)", "Mulch to prevent soil splash", "Avoid overhead watering", "Destroy crop debris after harvest"],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato Spider Mites",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Two-spotted spider mite (Tetranychus urticae) is a common pest causing stippling and bronzing of tomato leaves.",
        "symptoms": ["Fine yellow stippling on upper leaf surface", "Fine webbing on leaf undersides", "Bronzing and browning of leaves", "Severe defoliation in hot, dry conditions"],
        "treatment": ["Apply miticides (abamectin, bifenazate, spiromesifen)", "Use insecticidal soap or neem oil for organic control", "Release predatory mites (Phytoseiidae)"],
        "prevention": ["Avoid water stress — mites thrive on drought-stressed plants", "Monitor undersides of leaves regularly", "Avoid over-application of broad-spectrum insecticides that kill mite predators"],
    },
    "Tomato___Target_Spot": {
        "display": "Tomato Target Spot",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Moderate",
        "description": "Caused by Corynespora cassiicola, target spot affects leaves, stems, and fruit of tomatoes.",
        "symptoms": ["Brown lesions with concentric rings on leaves", "Dark lesions with water-soaked margins", "Fruit spots with depressed centres", "Defoliation in severe cases"],
        "treatment": ["Apply fungicides (azoxystrobin, chlorothalonil, mancozeb)", "Improve canopy airflow"],
        "prevention": ["Crop rotation", "Stake plants and train for airflow", "Avoid overhead irrigation"],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "Critical",
        "description": "TYLCV is transmitted by the silverleaf whitefly (Bemisia tabaci) and can cause near-total crop loss.",
        "symptoms": ["Upward curling and yellowing of young leaves", "Stunted plant growth", "Flowers drop, few fruit set", "Small, mottled fruit"],
        "treatment": ["No cure; remove and destroy infected plants early", "Control whitefly populations with insecticides (imidacloprid, thiamethoxam)", "Use yellow sticky traps to monitor whitefly levels"],
        "prevention": ["Plant TYLCV-resistant varieties", "Use reflective mulch to repel whiteflies", "Install insect-proof netting in nurseries", "Remove weeds that host whiteflies"],
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato Mosaic Virus",
        "plant": "Tomato",
        "status": "diseased",
        "severity": "High",
        "description": "Tomato mosaic virus (ToMV) is mechanically transmitted and can persist in soil and on tools.",
        "symptoms": ["Mosaic pattern of light and dark green on leaves", "Distorted, 'fernleaf' appearance", "Stunted growth", "Mottled, uneven fruit"],
        "treatment": ["No cure; remove and destroy infected plants", "Disinfect all tools with 10% bleach solution", "Control aphid vectors"],
        "prevention": ["Use virus-free certified seed", "Plant resistant varieties", "Wash hands before working with plants", "Avoid tobacco products near plants (cross-contamination risk)"],
    },
    "Tomato___healthy": {
        "display": "Healthy Tomato",
        "plant": "Tomato",
        "status": "healthy",
        "severity": "None",
        "description": "Your tomato plant appears healthy with no visible signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Stake or cage plants for support", "Mulch to conserve moisture and prevent splash", "Water consistently at the base", "Scout weekly for early disease or pest signs"],
    },
}


def predict_disease(image_bytes):
    """Load image, preprocess, and predict disease class."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    preds = get_model().predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    top_class = CLASS_NAMES[top_idx]

    # Top-3 predictions
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"label": CLASS_NAMES[i], "display": DISEASE_INFO.get(CLASS_NAMES[i], {}).get("display", CLASS_NAMES[i]), "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]

    info = DISEASE_INFO.get(top_class, {
        "display": top_class.replace("___", " — ").replace("_", " "),
        "plant": "Unknown",
        "status": "unknown",
        "severity": "Unknown",
        "description": "No detailed information available for this condition.",
        "symptoms": [],
        "treatment": [],
        "prevention": [],
    })

    return {
        "class": top_class,
        "confidence": round(top_conf * 100, 2),
        "top3": top3,
        "info": info,
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {"png", "jpg", "jpeg", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or WEBP."}), 400

    image_bytes = file.read()

    # Encode image for preview in result
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = f"image/{ext if ext != 'jpg' else 'jpeg'}"

    result = predict_disease(image_bytes)
    result["image_data"] = f"data:{mime};base64,{b64}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
