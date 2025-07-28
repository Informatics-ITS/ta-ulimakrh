import streamlit as st
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATHS = {
    # Augmented models
    'KNN TF-IDF + Augmentasi': r"C:\Users\admin\Downloads\MODEL\knn_tfidf_aug.pkl",
    'KNN Word2VEC + Augmentasi': r"C:\Users\admin\Downloads\MODEL\knn_w2v_aug.pkl",
    'MLP TF-IDF + Augmentasi': r"C:\Users\admin\Downloads\MODEL\mlp_tfidf_aug.pkl",
    'MLP Word2VEC + Augmentasi': r"C:\Users\admin\Downloads\MODEL\mlp_w2v_aug.pkl",
    
    # Non-Augmented models
    'KNN TF-IDF': r"C:\Users\admin\Downloads\MODEL\knn_tfidf.pkl",
    'KNN Word2VEC': r"C:\Users\admin\Downloads\MODEL\knn_w2v.pkl",
    'MLP TF-IDF': r"C:\Users\admin\Downloads\MODEL\mlp_tfidf.pkl",
    'MLP Word2VEC': r"C:\Users\admin\Downloads\MODEL\mlp_w2v.pkl",
}

VECTORIZER_PATHS = {
    'aug': r'C:\Users\admin\Downloads\MODEL\tfidf_vectorizer_aug.pkl',
    'raw': r'C:\Users\admin\Downloads\MODEL\tfidf_raw_vectorizer.pkl'
}

WORD2VEC_PATHS = {
    'aug': r'C:\Users\admin\Downloads\MODEL\word2vec_aug.model',
    'raw': r'C:\Users\admin\Downloads\MODEL\word2vec_raw.model'
}

MBTI_TYPES = {
    0: "ENFJ: The Protagonist",
    1: "ENFP: The Campaigner", 
    2: "ENTJ: The Commander",
    3: "ENTP: The Debater",
    4: "ESFJ: The Consul",
    5: "ESFP: The Entertainer",
    6: "ESTJ: The Executive",
    7: "ESTP: The Entrepreneur",
    8: "INFJ: The Advocate",
    9: "INFP: The Mediator",
    10: "INTJ: The Architect",
    11: "INTP: The Thinker",
    12: "ISFJ: The Defender",
    13: "ISFP: The Adventurer",
    14: "ISTJ: The Logistician",
    15: "ISTP: The Virtuoso"
}

@st.cache_resource
def load_model(file_path, model_type="generic"):
    """Load model with caching and error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error loading {model_type} model from {file_path}: {str(e)}")
        st.error(f"Failed to load {model_type} model: {str(e)}")
        return None

@st.cache_resource
def load_word2vec_model(file_path):
    """Load Word2Vec model with caching and error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Word2Vec model file not found: {file_path}")
        return Word2Vec.load(file_path)
    except Exception as e:
        logger.error(f"Error loading Word2Vec model from {file_path}: {str(e)}")
        st.error(f"Failed to load Word2Vec model: {str(e)}")
        return None

def validate_input(input_text):
    """Validate user input"""
    if not input_text or not input_text.strip():
        return False, "Please enter some text to analyze."
    
    if len(input_text.strip()) < 10:
        return False, "Please enter at least 10 characters for better analysis."
    
    return True, ""

def evaluate_knn_or_mlp(model, X_test, model_type="KNN"):
    """Improved evaluation function with better error handling"""
    try:
        if model is None:
            return None, 0.0
            
        y_pred = model.predict(X_test)
        
        if model_type == "KNN":
            # Untuk KNN, gunakan jarak ke tetangga terdekat
            distances, indices = model.kneighbors(X_test)
            # Normalisasi confidence score antara 0-1
            max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
            confidence_score = 1 - (np.min(distances[0]) / (max_distance + 1e-10))
            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp ke [0,1]
            
        else:  # MLP
            # Untuk MLP, gunakan probabilitas prediksi
            y_prob = model.predict_proba(X_test)
            confidence_score = float(np.max(y_prob))
        
        return y_pred, confidence_score
        
    except Exception as e:
        logger.error(f"Error in evaluation for {model_type}: {str(e)}")
        st.error(f"Error evaluating {model_type} model: {str(e)}")
        return None, 0.0

def get_consensus(predictions):
    """Get consensus prediction with validation"""
    if not predictions:
        return None
    
    valid_predictions = [p for p in predictions if p is not None]
    if not valid_predictions:
        return None
        
    return Counter(valid_predictions).most_common(1)[0][0]

def get_mbti_info(prediction):
    """Get MBTI information with validation"""
    if prediction is None:
        return "Unable to determine MBTI type"
    return MBTI_TYPES.get(prediction, f"Unknown MBTI Type (Code: {prediction})")

def preprocess_input(input_text, vectorizer=None, word2vec_model=None):
    """Improved preprocessing with error handling"""
    try:
        if vectorizer:
            return vectorizer.transform([input_text])
        elif word2vec_model:
            words = input_text.lower().split()  # Convert to lowercase
            vector = np.zeros(word2vec_model.vector_size)
            word_count = 0
            
            for word in words:
                if word in word2vec_model.wv:
                    vector += word2vec_model.wv[word]
                    word_count += 1
            
            if word_count > 0:
                vector /= word_count
            else:
                logger.warning("No words found in Word2Vec vocabulary")
                
            return np.array([vector])
        else:
            return input_text
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        st.error(f"Error processing input text: {str(e)}")
        return None

def get_model_and_preprocessor(model_choice):
    """Get model and appropriate preprocessor"""
    try:
        # Load model
        model_path = MODEL_PATHS.get(model_choice)
        if not model_path:
            raise ValueError(f"Unknown model choice: {model_choice}")
        
        model_type = "KNN" if 'KNN' in model_choice else "MLP"
        model = load_model(model_path, model_type)
        
        # Determine preprocessor type
        is_augmented = 'Augmentasi' in model_choice
        is_word2vec = 'Word2VEC' in model_choice
        
        if is_word2vec:
            path_key = 'aug' if is_augmented else 'raw'
            preprocessor = load_word2vec_model(WORD2VEC_PATHS[path_key])
        else:
            path_key = 'aug' if is_augmented else 'raw'
            preprocessor = load_model(VECTORIZER_PATHS[path_key], "TF-IDF Vectorizer")
        
        return model, preprocessor, is_word2vec
        
    except Exception as e:
        logger.error(f"Error loading model {model_choice}: {str(e)}")
        st.error(f"Error loading model {model_choice}: {str(e)}")
        return None, None, False

# Streamlit Interface
st.title('🧠 MBTI Personality Type Predictor')
st.markdown("Enter some text in English and let our AI models predict your MBTI personality type!")

# Input validation and UI improvements
input_text = st.text_area(
    "Input text in English:", 
    placeholder="Write something about yourself, your thoughts, or experiences...",
    height=150
)

# Show character count
if input_text:
    st.caption(f"Character count: {len(input_text)}")

# Tombol untuk memulai evaluasi
if st.button('🔍 Analyze Personality Type', type="primary"):
    # Validate input
    is_valid, error_message = validate_input(input_text)
    if not is_valid:
        st.error(error_message)
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model_choices = [
        'KNN TF-IDF + Augmentasi',
        'KNN Word2VEC + Augmentasi', 
        'MLP TF-IDF + Augmentasi',
        'MLP Word2VEC + Augmentasi',
        'KNN TF-IDF',
        'KNN Word2VEC',
        'MLP TF-IDF',
        'MLP Word2VEC'
    ]
    
    all_predictions = {}
    successful_predictions = []
    
    for i, model_choice in enumerate(model_choices):
        status_text.text(f"Processing {model_choice}...")
        progress_bar.progress((i + 1) / len(model_choices))
        
        # Get model and preprocessor
        model, preprocessor, is_word2vec = get_model_and_preprocessor(model_choice)
        
        if model is None or preprocessor is None:
            st.warning(f"Skipping {model_choice} due to loading error")
            continue
        
        # Preprocess input
        if is_word2vec:
            X_test = preprocess_input(input_text, word2vec_model=preprocessor)
        else:
            X_test = preprocess_input(input_text, vectorizer=preprocessor)
        
        if X_test is None:
            st.warning(f"Skipping {model_choice} due to preprocessing error")
            continue
        
        # Evaluate model
        model_type = "KNN" if 'KNN' in model_choice else "MLP"
        prediction, confidence_score = evaluate_knn_or_mlp(model, X_test, model_type)
        
        if prediction is not None:
            all_predictions[model_choice] = {
                "prediction": prediction[0],
                "confidence": confidence_score
            }
            successful_predictions.append(prediction[0])
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if not all_predictions:
        st.error("❌ No models could process your input successfully. Please try again.")
        st.stop()
    
    st.success(f"✅ Successfully analyzed with {len(all_predictions)} models!")
    
    # Show individual model results
    st.subheader("📊 Individual Model Results")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    for i, (model_choice, result) in enumerate(all_predictions.items()):
        with col1 if i % 2 == 0 else col2:
            with st.expander(f"🤖 {model_choice}"):
                mbti_info = get_mbti_info(result['prediction'])
                st.write(f"**MBTI Type:** {mbti_info}")
                st.write(f"**Confidence:** {result['confidence']:.3f}")
                
                # Confidence bar
                confidence_percentage = result['confidence'] * 100
                st.progress(result['confidence'])
                st.caption(f"{confidence_percentage:.1f}% confident")
    
    # Consensus prediction
    if len(successful_predictions) > 1:
        consensus = get_consensus(successful_predictions)
        
        st.subheader("🎯 Final Prediction (Consensus)")
        if consensus is not None:
            mbti_info = get_mbti_info(consensus)
            
            # Create a nice display for the final result
            st.success(f"**Your predicted MBTI type is: {mbti_info}**")
            
            # Show voting breakdown
            vote_counts = Counter(successful_predictions)
            st.write("**Voting breakdown:**")
            for prediction, count in vote_counts.most_common():
                mbti_name = get_mbti_info(prediction)
                percentage = (count / len(successful_predictions)) * 100
                st.write(f"- {mbti_name}: {count}/{len(successful_predictions)} votes ({percentage:.1f}%)")
        else:
            st.error("Unable to determine consensus prediction")
    else:
        st.info("Only one model prediction available - no consensus needed")

# Add information sidebar
with st.sidebar:
    st.header("ℹ️ About MBTI")
    st.write("""
    The Myers-Briggs Type Indicator (MBTI) is a personality assessment that categorizes people into 16 different personality types based on four key dimensions:
    
    - **E/I**: Extraversion vs Introversion
    - **S/N**: Sensing vs Intuition  
    - **T/F**: Thinking vs Feeling
    - **J/P**: Judging vs Perceiving
    """)
    
    st.header("🤖 Models Used")
    st.write("""
    This app uses 8 different model scenarios:
    - **KNN**: K-Nearest Neighbors
    - **MLP**: Multi-Layer Perceptron
    - **TF-IDF**: Term Frequency-Inverse Document Frequency
    - **Word2Vec**: Word embedding vectors
    - **Augmentation**: Data augmentation techniques
    """)
    
    st.header("💡 Tips")
    st.write("""
    For better results:
    - Write at least 50-100 words
    - Be descriptive about your thoughts and preferences
    - Use natural language
    - Include personal examples
    """)