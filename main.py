import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def apply_distortion(image, distortion_level, distortion_profile):
    height, width = image.shape[:2]
    distorted_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            normalized_x = (x / width) * 2 - 1
            normalized_y = (y / height) * 2 - 1

            radius = np.sqrt(normalized_x**2 + normalized_y**2)
            theta = np.arctan2(normalized_y, normalized_x)

            profile_adjusted_radius = radius ** (1 + distortion_profile)

            if distortion_level >= 0:  # Barrel distortion
                distorted_radius = profile_adjusted_radius * (1 + distortion_level * profile_adjusted_radius**2)
            else:  # Pincushion distortion
                distorted_radius = profile_adjusted_radius / (1 - distortion_level * profile_adjusted_radius**2)

            distorted_x = ((distorted_radius * np.cos(theta) + 1) * width) / 2
            distorted_y = ((distorted_radius * np.sin(theta) + 1) * height) / 2

            if 0 <= distorted_x < width and 0 <= distorted_y < height:
                distorted_image[y, x] = image[int(distorted_y), int(distorted_x)]

    return distorted_image


def get_eye_center(landmarks, eye_indices):
    x = sum(landmarks[landmark].x for landmark in eye_indices) / len(eye_indices)
    y = sum(landmarks[landmark].y for landmark in eye_indices) / len(eye_indices)
    return np.array([x, y])

def get_key_points(face_landmarks, image_shape):
    right_eye_indices = [133, 134, 153, 154]
    left_eye_indices = [362, 363, 382, 383]

    right_eye_center = get_eye_center(face_landmarks.landmark, right_eye_indices)
    left_eye_center = get_eye_center(face_landmarks.landmark, left_eye_indices)

    right_eye_center_scaled = right_eye_center * image_shape[1::-1]
    left_eye_center_scaled = left_eye_center * image_shape[1::-1]

    mouth_center = np.array([face_landmarks.landmark[13].x, face_landmarks.landmark[13].y]) * image_shape[1::-1]

    return np.array([right_eye_center_scaled, left_eye_center_scaled, mouth_center], dtype=np.float32)

def draw_face_mesh(image, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    return image

def align_images(image1, image2, draw_mesh):
    results1 = face_mesh.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    results2 = face_mesh.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    if results1.multi_face_landmarks and results2.multi_face_landmarks:
        keypoints1 = get_key_points(results1.multi_face_landmarks[0], image1.shape)
        keypoints2 = get_key_points(results2.multi_face_landmarks[0], image2.shape)

        M = cv2.getAffineTransform(keypoints2, keypoints1)
        aligned_image2 = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

        results2_aligned = face_mesh.process(cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2RGB))

        if draw_mesh:
            aligned_image1 = draw_face_mesh(image1, results1)
            aligned_image2 = draw_face_mesh(aligned_image2, results2_aligned)
        else:
            aligned_image1 = image1
            aligned_image2 = aligned_image2

        return aligned_image1, aligned_image2

    return image1, image2

st.title("Test")

draw_mesh = st.sidebar.checkbox("Mesh", True)

image_path1 = '1.jpg'
image_path2 = '2.jpg'

image1 = cv2.cvtColor(cv2.imread(image_path1), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2RGB)



st.sidebar.title("Img 1")
distortion_level1 = st.sidebar.slider("Intensity1", -0.5, 0.5, 0.0)
distortion_profile1 = st.sidebar.slider("Decay1", -1.0, 1.0, 0.0)

st.sidebar.title("Img 2")
distortion_level2 = st.sidebar.slider("Intensity2", -0.5, 0.5, 0.0)
distortion_profile2 = st.sidebar.slider("Decay2", -1.0, 1.0, 0.0)

distorted_image1 = apply_distortion(image1, distortion_level1, distortion_profile1)
distorted_image2 = apply_distortion(image2, distortion_level2, distortion_profile2)

aligned_image1, aligned_image2 = align_images(distorted_image1, distorted_image2, draw_mesh)

col1, col2 = st.columns(2)
with col1:
    st.image(aligned_image1, caption='Img 1', use_column_width=True)
with col2:
    st.image(aligned_image2, caption='Img 2', use_column_width=True)

face_mesh.close()
