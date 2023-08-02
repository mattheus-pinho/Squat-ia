import os
from tkinter import ttk, filedialog
import cv2
import mediapipe as mp
import numpy as np
import csv
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk

#AI Functions
def draw_lines(frame, results, mp_pose):
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_point = (
            int(results.pose_landmarks.landmark[start_idx].x * frame.shape[1]),
            int(results.pose_landmarks.landmark[start_idx].y * frame.shape[0]),
        )
        end_point = (
            int(results.pose_landmarks.landmark[end_idx].x * frame.shape[1]),
            int(results.pose_landmarks.landmark[end_idx].y * frame.shape[0]),
        )
        color = (0, 255, 0)  # Verde para todas as linhas, exceto a perna esquerda
        if (
            start_idx == mp_pose.PoseLandmark.LEFT_KNEE
            or end_idx == mp_pose.PoseLandmark.LEFT_KNEE
        ):
            color = (0, 0, 255)  # Vermelho para a perna esquerda
        cv2.line(frame, start_point, end_point, color, 2)

def collect_training_data(cap):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    training_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            draw_lines(frame, results, mp_pose)

            left_knee = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                    * frame.shape[0]
                ),
            )
            left_ankle = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
                    * frame.shape[0]
                ),
            )

            angle = np.degrees(
                np.arctan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0])
            )
            training_data.append(round(angle, 4))

            cv2.putText(
                frame,
                "Agachamento Correto",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Squat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return training_data

def save_training_data(data, file_path):
    with open(
        file_path, "a", newline=""
    ) as csvfile:  # Modo "a" para adicionar novas linhas
        writer = csv.writer(csvfile)
        for angle in data:
            writer.writerow([angle])

def load_svm_model(svm_model_path):
    return joblib.load(svm_model_path)

def analyze_video(video_path, svm_model):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            draw_lines(frame, results, mp_pose)

            left_knee = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                    * frame.shape[0]
                ),
            )
            left_ankle = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
                    * frame.shape[0]
                ),
            )

            angle = np.degrees(
                np.arctan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0])
            )
            is_correct_squat = svm_model.predict([[angle]])

            cv2.putText(
                frame,
                "Agachamento Correto"
                if is_correct_squat[0] == 1
                else "Agachamento Incorreto",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if is_correct_squat[0] == 1 else (0, 0, 255),
                2,
            )

            cv2.imshow("Squat Analysis", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#AI functions used by GUI
def real_time_cam():
    cap = cv2.VideoCapture(0)
    training_data = collect_training_data(cap)
    save_training_data(training_data, "training_data.csv")

def training_video():
    video_path = open_file_dialog()
    cap = cv2.VideoCapture(video_path)
    training_data = collect_training_data(cap)
    save_training_data(training_data, "training_data.csv")

def analyze_client_video():
    video_path = open_file_dialog()
    svm_model_path = "modelo_svm_agachamento.pkl"
    svm_model = load_svm_model(svm_model_path)
    analyze_video(video_path, svm_model)

#GUI Functions
def show_tab(tab_index):
    tab_control.select(tab_index)

def create_tab_button(tab_control, tab_index, button_text):
    button_frame = tab_control.winfo_children()[tab_index]
    button = tk.Button(button_frame, text=button_text, command=lambda: show_tab(tab_index))
    button.pack(padx=20, pady=10)
    close_button = tk.Button(button_frame, text="Close", command=window.destroy)
    close_button.pack(padx=20, pady=5)

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Selecione um arquivo")
    if file_path:
        absolute_path = os.path.abspath(file_path)
        return absolute_path

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Tab Example")
    window.geometry("800x600")

    tab_control = ttk.Notebook(window)

    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)
    tab4 = ttk.Frame(tab_control)

    tab_control.add(tab1, text="Tab 1")
    tab_control.add(tab2, text="Tab 2")
    tab_control.add(tab3, text="Tab 3")
    tab_control.add(tab4, text="Tab 4")

    #tab1
    title_tab1 = tk.Label(tab1, text="Usar câmera em tempo real para treinar o modelo")
    title_tab1.pack(padx=20, pady=10)

    button_tab1 = tk.Button(tab1, text="Iniciar camera", command=lambda: real_time_cam())
    button_tab1.pack(padx=20, pady=10)

    #tab2
    title_tab2 = tk.Label(tab2, text="Usar vídeo de treinamento")
    title_tab2.pack(padx=20, pady=20)

    open_button = tk.Button(tab2, text="Abrir Arquivo", command=training_video)
    open_button.pack(padx=20, pady=10)

    #tab3


    #tab4

    #Fim das Tabs
    tab_control.pack(expand=1, fill="both")

    window.mainloop()

    print("Escolha uma opção:")
    print("1 - ")
    print("2 - ")
    print("3 - Análise de agachamentos em novos dados")
    print("4 - Análise de agachamentos em dados da webcam")

    option = int(input("Digite o número da opção desejada: "))

    #testar a funcionalidade(Num 4)
    #svm_model_path = "modelo_svm_agachamento.pkl"
    #svm_model = load_svm_model(svm_model_path)
    #analyze_video(0, svm_model)  # Use 0 para a webcam