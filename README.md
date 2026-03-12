# MemeMirror
Point your webcam, make a face, and MemeMirror instantly matches your expression to a meme. Built in 24 hours at the Silicon Valley Hackathon using MediaPipe facial recognition to detect emotions (happy, sad, shocked, surprised, neutral) and display matching memes in under one second.
MemeMirror is a web application that uses computer vision and facial recognition to analyze your expressions through your webcam and instantly pair them with contextually relevant memes. Point your webcam, make a face, and watch as AI matches you to the perfect meme in under one second.
Features

📷 Real-time facial expression detection using MediaPipe Face Landmarker
🎭 Emotion classification (happy, sad) based on facial blendshapes
⚡ Sub-second latency from camera capture to meme display
📊 Debug signals view showing real-time facial metrics (smile, frown, jaw open, eyebrow raise)
🌙 Dark-mode responsive interface with live webcam feed
🔒 Privacy-first: All processing happens locally, no data stored

How It Works

Webcam Capture: JavaScript captures live frames using the getUserMedia API
Image Processing: Frames are converted to base64 and sent to Flask backend
Facial Analysis: MediaPipe analyzes 52 facial blendshapes (smile intensity, jaw movement, eyebrow position, frown depth)
Emotion Detection: Custom classification logic uses threshold-based detection with a 10-frame stability buffer
Meme Matching: Detected emotion maps to corresponding meme in database
Display: Matched meme appears instantly with optional debug metrics

Tech Stack
Backend:

Python 3.14
Flask (web server)
MediaPipe (facial landmark detection)
OpenCV (image processing)
NumPy (numerical operations)

Frontend:

Vanilla JavaScript
HTML5/CSS3
WebRTC getUserMedia API

Installation
Prerequisites

Python 3.8 or higher
Webcam
