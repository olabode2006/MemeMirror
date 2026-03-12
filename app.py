
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import base64
import numpy as np
import random

app = Flask(__name__, static_folder='static')
CORS(app)

# Stability buffer
emotion_buffer = deque(maxlen=10)

# Create FaceLandmarker
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1,
)

detector = vision.FaceLandmarker.create_from_options(options)

# Meme database - maps emotions to memes
MEME_DATABASE = {
    "happy": {
        "meme_id": "big_smile",
        "meme_name": "Big Smile",
        "image": "/static/memes/big_smile.jpg",
    },
    "shocked": {
        "meme_id": "surprised_pikachu",
        "meme_name": "Surprised Pikachu",
        "image": "/static/memes/surprised_pikachu.jpg",
    },
    "surprised": {
        "meme_id": "surprised_pikachu",
        "meme_name": "Surprised Pikachu",
        "image": "/static/memes/surprised_pikachu.jpg",
    },
    "sad": {
        "meme_id": "tired_face",
        "meme_name": "Tired Face",
        "image": "/static/memes/tired_face.jpg",
    },
    "neutral": {
        "meme_id": "tired_face",
        "meme_name": "Tired Face",
        "image": "/static/memes/tired_face.jpg",
    }
}

# Caption database by tone
CAPTIONS = {
    "delulu": [
        "I'm literally the main character of this story 💅",
        "Everything always works out for me because I'm just built different ✨",
        "My delusion is my reality and honestly? It's working 🌟",
        "If manifestation isn't real then explain how I got here 🦋",
        "Living in my own little world and the rent is free 🏰",
        "My therapist said delusion is unhealthy but look at me thriving 💫",
        "Reality is whatever I decide it is today 🌈",
        "The voices in my head all agree I'm doing amazing 🎭",
        "Gaslight, gatekeep, girlboss but make it delulu 👑",
        "I'm not delusional, I'm just living in a different timeline ⏰"
    ],
    "brutal": [
        "You thought that was going to work? Really? 💀",
        "Not you thinking you did something there 😭",
        "The audacity to exist like this is actually impressive",
        "I would roast you but life already did that",
        "This is what happens when confidence exceeds ability",
        "Please tell me this is satire because yikes",
        "I've seen NFTs with more value than this take",
        "The bar was on the floor and you still tripped over it",
        "I'm not saying you're wrong but actually yes I am",
        "This ain't it chief and it will never be it"
    ],
    "corporate": [
        "Let's circle back on this initiative in Q4 📊",
        "Synergizing our core competencies for maximum ROI 💼",
        "Moving the needle on KPIs through strategic alignment 📈",
        "Leveraging best practices to optimize our bandwidth allocation",
        "This aligns with our mission to create shareholder value 💰",
        "Let's touch base offline to discuss actionable insights",
        "Pivoting our strategy to capture emerging market opportunities",
        "Ideating solutions that move us up and to the right 📉➡️📈",
        "Operationalizing our vision through agile methodologies",
        "Fostering innovation within our dynamic ecosystem 🚀"
    ],
    "faith": [
        "God's plan is unfolding exactly as it should 🙏",
        "Sometimes the test comes before the testimony ✝️",
        "Walking by faith, not by sight 🕊️",
        "The struggle you're in today is developing the strength you need tomorrow",
        "Let go and let God handle what you can't control 🌟",
        "Your current situation is not your final destination",
        "Faith over fear, always and forever 💫",
        "He doesn't give you the people you want, He gives you the people you need",
        "In the middle of difficulty lies opportunity - trust the process 🙌",
        "Every setback is a setup for a comeback through Him ⚡"
    ]
}


def classify_emotion(blendshapes):
    """Classify emotion based on facial blendshapes"""
    emotion = "neutral"
    confidence = 0.0

    smile = (
        blendshapes.get("mouthSmileLeft", 0) + blendshapes.get("mouthSmileRight", 0)
    ) / 2

    jaw_open = blendshapes.get("jawOpen", 0)
    brow_up = blendshapes.get("browInnerUp", 0)
    frown = (
        blendshapes.get("mouthFrownLeft", 0) + blendshapes.get("mouthFrownRight", 0)
    ) / 2

    if smile > 0.6:
        emotion = "happy"
        confidence = smile
    elif jaw_open > 0.6 and brow_up > 0.3:
        emotion = "shocked"
        confidence = jaw_open
    elif brow_up > 0.5:
        emotion = "surprised"
        confidence = brow_up
    elif frown > 0.5:
        emotion = "sad"
        confidence = frown

    return emotion, confidence


@app.route('/')
def index():
    return send_from_directory('.', 'MemeMirror.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze facial expression from camera frame"""
    try:
        data = request.get_json()
        
        # Get base64 image from frontend
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face and get blendshapes
        result = detector.detect(mp_image)
        
        emotion = "neutral"
        confidence = 0.0
        signals = {}
        
        if result.face_blendshapes:
            blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
            emotion, confidence = classify_emotion(blendshapes)
            
            # Extract key signals for display
            signals = {
                "smile": round((blendshapes.get("mouthSmileLeft", 0) + blendshapes.get("mouthSmileRight", 0)) / 2, 2),
                "mouth_open": round(blendshapes.get("jawOpen", 0), 2),
                "eyebrow_raise": round(blendshapes.get("browInnerUp", 0), 2),
                "frown": round((blendshapes.get("mouthFrownLeft", 0) + blendshapes.get("mouthFrownRight", 0)) / 2, 2)
            }
        
        # Add to stability buffer
        emotion_buffer.append(emotion)
        stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        
        # Get matching meme
        meme_data = MEME_DATABASE.get(stable_emotion, MEME_DATABASE["sad"])
        
        response = {
            "emotion": stable_emotion,
            "confidence": round(confidence, 2),
            "score": round(confidence, 2),
            "meme_id": meme_data["meme_id"],
            "meme_name": meme_data["meme_name"],
            "image_url": meme_data["image"],
            "signals": signals
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/caption', methods=['POST'])
def generate_caption():
    """Generate captions based on tone"""
    try:
        data = request.get_json()
        tone = data.get('tone', 'delulu').lower()
        
        # Get captions for the specified tone
        caption_pool = CAPTIONS.get(tone, CAPTIONS['delulu'])
        
        # Randomly select 3 captions
        selected_captions = random.sample(caption_pool, min(3, len(caption_pool)))
        
        return jsonify({
            "captions": selected_captions,
            "tone": tone
        })
        
    except Exception as e:
        print(f"Error in generate_caption: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
