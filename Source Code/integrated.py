import numpy as np
import librosa
from keras.models import load_model
import pyaudio
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image

# Function to extract features from an audio file
def extract_features(audio_data, sample_rate, mfcc=True, chroma=True, mel=True):
    if chroma:
        stft = np.abs(librosa.stft(audio_data))
    result = []
    if mfcc:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        result.append(mfccs_mean)
        print("Extracted MFCCs:", mfccs_mean)
    if chroma:
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        result.append(chroma_mean)
        print("Extracted Chroma:", chroma_mean)
    if mel:
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_mean = np.mean(mel, axis=1)
        result.append(mel_mean)
        print("Extracted Mel Spectrogram:", mel_mean)
    max_len = max(feature.shape[0] for feature in result)
    result = [np.pad(feature, (0, max_len - feature.shape[0])) if feature.shape[0] < max_len else feature[:max_len] for feature in result]
    return np.concatenate(result, axis=0)

# Load the trained model for voice emotion detection
voice_model = load_model('D:\\D\\Sudharsan\\Mini project\\voice_emotion_cnn_lstm.h5')
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to record audio from microphone
def record_audio(duration=5, sample_rate=44100, channels=1, chunk=1024, input_device_index=None):
    audio = pyaudio.PyAudio()
    
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        print(f"Device {i}: {audio.get_device_info_by_index(i)['name']}")
    
    stream = audio.open(format=pyaudio.paInt16, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk,
                        input_device_index=input_device_index)
    
    print("Recording...")
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        try:
            data = stream.read(chunk)
            frames.append(data)
        except Exception as e:
            print("Error recording audio:", e)
            break
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    audio_data = b''.join(frames)
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    
    return audio_data, sample_rate

# Function to predict emotion from audio data and get probabilities
def predict_voice_emotion(audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate, mfcc=True, chroma=True, mel=True)
    max_sequence_length = 100
    features = np.resize(features, (1, max_sequence_length, 1))  # Reshape for Conv1D input
    prediction = voice_model.predict(features)[0]  # Get the probabilities for each class
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = observed_emotions[predicted_emotion_index]
    return predicted_emotion, prediction

# Function to capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)  # Access the default camera (index 0)
    
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    return frame

# Function to predict emotion from an image
def predict_face_emotion(image_path):
    # Load the face emotion prediction model
    face_model = load_model('D:\\D\\Sudharsan\\Mini project\\face emotion recoginition.h5')
    
    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Make the prediction
    prediction = face_model.predict(img_array)
    
    # Define emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Get the predicted emotion index and label
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    
    return predicted_emotion, prediction[0]

def main():
    # Capture image from camera
    captured_image = capture_image()
    
    if captured_image is not None:
        # Resize the captured image to 224x224 pixels
        resized_image = Image.fromarray(captured_image).resize((224, 224))
        resized_image_path = "captured_image_resized.jpg"
        resized_image.save(resized_image_path)
        
        # Predict emotion from the resized image
        predicted_face_emotion, face_probabilities = predict_face_emotion(resized_image_path)
        print("Predicted Face Emotion:", predicted_face_emotion)
        print("Face Emotion Probabilities:")
        for emotion, probability in zip(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], face_probabilities):
            print(f"Probability of {emotion}: {probability:.5f}")
    else:
        print("Error: Unable to capture image from camera")
        return  # Exit the function if the image cannot be captured
    
    # Record audio from microphone
    duration = 5
    input_device_index = 0
    audio_data, sample_rate = record_audio(duration=duration, input_device_index=input_device_index)
    
    # Predict emotion from the recorded audio
    predicted_voice_emotion, voice_emotion_probabilities = predict_voice_emotion(audio_data, sample_rate)
    print("\nPredicted Voice Emotion:", predicted_voice_emotion)
    print("Voice Emotion Probabilities:")
    for emotion, probability in zip(observed_emotions, voice_emotion_probabilities):
        print(f"Probability of {emotion}: {probability:.5f}")
    
    # Remove one emotion from voice emotion probabilities to match the length of face_probabilities
    voice_emotion_probabilities = np.delete(voice_emotion_probabilities, np.argmin(voice_emotion_probabilities))
    
    # Combine the probabilities by normalizing both arrays to the same length
    min_len = min(len(face_probabilities), len(voice_emotion_probabilities))
    
    # Normalize face_probabilities
    norm_face_probabilities = face_probabilities / np.sum(face_probabilities)
    
    # Normalize voice_emotion_probabilities and match length to face_probabilities
    norm_voice_probabilities = voice_emotion_probabilities / np.sum(voice_emotion_probabilities)
    
    # Combine probabilities by averaging
    combined_probabilities = (norm_face_probabilities + norm_voice_probabilities) / 2
    
    # Find the emotion with the highest combined probability
    max_prob_index = np.argmax(combined_probabilities)
    predicted_emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][max_prob_index]
    
    print("\nPredicted Combined Emotion:", predicted_emotion)
    print("Combined Emotion Probabilities:")
    for emotion, probability in zip(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], combined_probabilities):
        print(f"Probability of {emotion}: {probability:.5f}")

if __name__ == "__main__":
    main()
