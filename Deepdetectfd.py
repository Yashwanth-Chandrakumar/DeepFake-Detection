import gradio as gr
import cv2
import numpy as np
from tensorflow import keras
import face_recognition

# Load the pre-trained model
model3 = keras.models.load_model("deepdetectv4.h5")

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

    # Loop over frames
    while True:
        # Read a frame from the video
        ret, frame = video_input.read()

        # Break if no more frames
        if not ret:
            break

        # Detect faces in the frame using face_recognition
        face_locations = face_recognition.face_locations(frame)

        # Loop over detected faces
        for (top, right, bottom, left) in face_locations:
            # Preprocess the facial area
            face_img = cv2.resize(frame[top:bottom, left:right], (150, 150))
            face_img = np.expand_dims(face_img, axis=0) / 255.0
            prediction_probabilities = model3.predict(face_img)
            value = np.argmax(prediction_probabilities)
            prediction = "Real" if value == 1 else "Fake"
            print(prediction)

            # Draw a rectangle around the face and add text
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, prediction, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        video_output.write(frame)

    # Release the video capture and writer objects
    video_input.release()
    video_output.release()

    # Return the processed video file
    return output_filename

# Create the Gradio interface
iface = gr.Interface(process_video, gr.Video(), gr.File(file_count="multiple"))
iface.launch(share=True)
