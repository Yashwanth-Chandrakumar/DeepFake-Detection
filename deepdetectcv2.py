import gradio as gr
import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained model
model3 = keras.models.load_model("D:/Downloads/deepdetectv4.h5")

def process_video(video_file_path):
    # Open the video file
    video_input = cv2.VideoCapture(video_file_path)

    # Get video properties
    width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_input.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object for saving the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = 'processed_video.mp4'
    video_output = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop over frames
    while True:
        # Read a frame from the video
        ret, frame = video_input.read()

        # Break if no more frames
        if not ret:
            break

        # Detect faces in the frame using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over detected faces
        for (x, y, w, h) in faces:
            # Preprocess the facial area
            face_img = cv2.resize(frame[y:y+h, x:x+w], (150, 150))
            face_img = np.expand_dims(face_img, axis=0) / 255.0
            prediction_probabilities = model3.predict(face_img)
            value = np.argmax(prediction_probabilities)
            prediction = "Real" if value == 1 else "Fake"
            print(prediction)

            # Draw a rectangle around the face and add text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        video_output.write(frame)

    # Release the video capture and writer objects
    video_input.release()
    video_output.release()

    # Return the processed video file
    return output_filename

# Create the Gradio interface
iface = gr.Interface(process_video, gr.Video(), gr.File(file_count="multiple"))

# Launch the interface
iface.launch(share=True)
