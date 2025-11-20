# Celda 1
import cv2
import math
import numpy as np
import mediapipe as mp
import argparse
import os
from collections import deque
import time
import tkinter as tk
from tkinter import filedialog

# Configuración
CONFIRM_FRAMES = 3        # número de frames para confirmar un estado
SMOOTH_ALPHA = 0.6       # alfa para suavizado exponencial (0..1)
MIN_CONFIDENCE = 0.5     # umbral mínimo de MediaPipe para considerar landmark válido

# Rutas
ROBOT_IMG_DIR = "robot"  # carpeta con imágenes (asegúrate de tenerla)


# Celda 2
def dist(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

def angle_between(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def shoulder_to_wrist_angle(shoulder_px, wrist_px):
    sx, sy = shoulder_px
    wx, wy = wrist_px
    vx = wx - sx
    vy = sy - wy  # invertir Y para convención
    return math.degrees(math.atan2(vy, vx))

def exp_smooth(prev, new, alpha):
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev



# Celda 3 (REEMPLAZADA) - carga robusta de imágenes con normalización de nombres

STATE_IMAGES = {
    "both_down": "ambos_abajo.png",
    "both_up": "ambos_arriba.png",
    "both_extended": "ambos_extendida.png",
    "right_up_left_down": "derecha_arriba_izq_abajo.png",
    "right_up_left_extended": "derecha_arriba_izq_extendida.png",
    "right_extended_left_down": "derecha_extendida_izq_abajo.png",
    "left_up_right_down": "izquierda_arriba_der_abajo.png",
    "left_up_right_extended": "izquierda_arriba_der_extendida.png",
    "left_extended_right_down": "izquierda_extendida_der_abajo.png"
}

def _normalize_name(s: str) -> str:
    """
    Normaliza nombres: lowercase, reemplaza espacios y guiones por underscore,
    elimina múltiples underscores, quita acentos simples (opcional).
    """
    if s is None:
        return ""
    s = s.lower()
    # reemplazar acentos simples comunes (áéíóú -> aeiou)
    trans = str.maketrans("áéíóúüñ", "aeiouun")
    s = s.translate(trans)
    # reemplazar cualquier caracter no alfanumérico por underscore
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    # colapsar múltiples underscores
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s

def find_image_file(dirpath: str, expected_fname: str):
    """
    Busca en dirpath un archivo que, una vez normalizado, coincida con
    expected_fname normalizado. Retorna la ruta completa o None.
    """
    if not os.path.isdir(dirpath):
        return None
    expected_norm = _normalize_name(expected_fname)
    for fname in os.listdir(dirpath):
        # ignorar carpetas
        fpath = os.path.join(dirpath, fname)
        if not os.path.isfile(fpath):
            continue
        # extraer nombre base sin extension
        base, _ext = os.path.splitext(fname)
        if _normalize_name(base) == expected_norm:
            return fpath
    return None

def load_robot_images(dirpath=ROBOT_IMG_DIR):
    imgs = {}
    # Intenta cargar cada imagen usando el nombre exacto y si no busca una coincidencia normalizada
    if not os.path.isdir(dirpath):
        print(f"[WARN] carpeta de imágenes no existe: {dirpath}")
        # inicializar con None para que el resto del código funcione
        for state in STATE_IMAGES.keys():
            imgs[state] = None
        return imgs

    for state, fname in STATE_IMAGES.items():
        exact_path = os.path.join(dirpath, fname)
        if os.path.exists(exact_path):
            img = cv2.imread(exact_path, cv2.IMREAD_UNCHANGED)
            imgs[state] = img
        else:
            # Buscar coincidencia normalizada (por ejemplo "ambos arriba.png" -> "ambos_arriba.png")
            found = find_image_file(dirpath, fname)
            if found:
                img = cv2.imread(found, cv2.IMREAD_UNCHANGED)
                imgs[state] = img
                print(f"[INFO] {state}: encontrado '{found}' para '{fname}'")
            else:
                imgs[state] = None
                print(f"[WARN] falta imagen esperada para estado '{state}': buscado '{fname}'")
    return imgs

robot_images = load_robot_images()
# Diccionario auxiliar para cachear intentos de carga dinámica
_robot_load_cache = {}




# Celda 4
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_px(landmarks, width, height):
    mapping = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24
    }
    out = {}
    for name, idx in mapping.items():
        lm = landmarks[idx]
        px = int(lm.x * width)
        py = int(lm.y * height)
        out[name] = (px, py, lm.visibility)
    return out



# Celda 5
class HysteresisCounter:
    def __init__(self, confirm_frames=CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.counters = {}

    def update(self, state, is_true):
        cnt = self.counters.get(state, 0)
        if is_true:
            cnt = min(self.confirm_frames, cnt + 1)
        else:
            cnt = max(0, cnt - 1)
        self.counters[state] = cnt
        return cnt >= self.confirm_frames

hyst = HysteresisCounter()

smooth_store = {
    "left_elbow_angle": None,
    "right_elbow_angle": None,
    "left_shoulder_wrist_angle": None,
    "right_shoulder_wrist_angle": None,
    "left_hor_ratio": None,
    "right_hor_ratio": None
}

def compute_arm_features(px_should, px_elbow, px_wrist):
    elbow_angle = angle_between(px_should, px_elbow, px_wrist)
    sw_angle = shoulder_to_wrist_angle(px_should, px_wrist)
    arm_len = dist(px_should, px_elbow) + dist(px_elbow, px_wrist)
    hor_ratio = abs(px_wrist[0] - px_should[0]) / max(1.0, arm_len)
    return elbow_angle, sw_angle, hor_ratio

def classify_arm(elbow_angle, sw_angle, hor_ratio, shoulder_py, hip_py):
    if elbow_angle > 150 and hor_ratio > 0.55:
        return "extended"
    if sw_angle > 40:
        return "up"
    if sw_angle < -40 or shoulder_py > hip_py:
        return "down"
    if sw_angle > 15:
        return "up"
    if sw_angle < -15:
        return "down"
    return "unknown"



# Celda 6
def decide_state(left_state, right_state):
    if left_state == "down" and right_state == "down":
        return "both_down"
    if left_state == "up" and right_state == "up":
        return "both_up"
    if left_state == "extended" and right_state == "extended":
        return "both_extended"
    if right_state == "up" and left_state == "down":
        return "right_up_left_down"
    if right_state == "up" and left_state == "extended":
        return "right_up_left_extended"
    if right_state == "extended" and left_state == "down":
        return "right_extended_left_down"
    if left_state == "up" and right_state == "down":
        return "left_up_right_down"
    if left_state == "up" and right_state == "extended":
        return "left_up_right_extended"
    if left_state == "extended" and right_state == "down":
        return "left_extended_right_down"
    return "both_down"




# Celda 7 (REEMPLAZADA) - run() con carga dinámica de imágenes si no estaban cargadas antes

def _load_image_for_state_dynamic(state):
    """
    Si en robot_images no hay imagen para 'state', intenta encontrar y cargar
    dinámicamente (y la guarda en robot_images para próximas lecturas).
    """
    if state in _robot_load_cache:
        return _robot_load_cache[state]
    img = robot_images.get(state)
    if img is not None:
        _robot_load_cache[state] = img
        return img
    # intentar buscar archivo en carpeta con la misma lógica de normalización
    expected_fname = STATE_IMAGES.get(state)
    found = find_image_file(ROBOT_IMG_DIR, expected_fname) if expected_fname else None
    if found:
        img_loaded = cv2.imread(found, cv2.IMREAD_UNCHANGED)
        robot_images[state] = img_loaded  # cachear globalmente
        _robot_load_cache[state] = img_loaded
        print(f"[INFO] cargada dinámicamente para '{state}': {found}")
        return img_loaded
    _robot_load_cache[state] = None
    return None

def run(video_path):
    if not video_path or not os.path.exists(video_path):
        print("[ERROR] video_path inválido o no existe:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    mp_ctx = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    current_state = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o frame inválido.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_ctx.process(frame_rgb)

        vis = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(vis, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        left_state = right_state = "unknown"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            pts = landmarks_to_px(lm, width, height)

            lsx, lsy, lsv = pts["left_shoulder"]
            lex, ley, lev = pts["left_elbow"]
            lwx, lwy, lwv = pts["left_wrist"]
            rsx, rsy, rsv = pts["right_shoulder"]
            rex, rey, rev = pts["right_elbow"]
            rwx, rwy, rwv = pts["right_wrist"]
            lhipx, lhipy, _ = pts["left_hip"]
            rhipx, rhipy, _ = pts["right_hip"]

            if lsv > MIN_CONFIDENCE and lev > MIN_CONFIDENCE and lwv > MIN_CONFIDENCE:
                l_elbow_angle, l_sw_angle, l_hor_ratio = compute_arm_features(
                    (lsx, lsy), (lex, ley), (lwx, lwy)
                )
                smooth_store["left_elbow_angle"] = exp_smooth(smooth_store["left_elbow_angle"], l_elbow_angle, SMOOTH_ALPHA)
                smooth_store["left_shoulder_wrist_angle"] = exp_smooth(smooth_store["left_shoulder_wrist_angle"], l_sw_angle, SMOOTH_ALPHA)
                smooth_store["left_hor_ratio"] = exp_smooth(smooth_store["left_hor_ratio"], l_hor_ratio, SMOOTH_ALPHA)

                left_state = classify_arm(
                    smooth_store["left_elbow_angle"],
                    smooth_store["left_shoulder_wrist_angle"],
                    smooth_store["left_hor_ratio"],
                    lsy, min(lhipy, rhipy)
                )

            if rsv > MIN_CONFIDENCE and rev > MIN_CONFIDENCE and rwv > MIN_CONFIDENCE:
                r_elbow_angle, r_sw_angle, r_hor_ratio = compute_arm_features(
                    (rsx, rsy), (rex, rey), (rwx, rwy)
                )
                smooth_store["right_elbow_angle"] = exp_smooth(smooth_store["right_elbow_angle"], r_elbow_angle, SMOOTH_ALPHA)
                smooth_store["right_shoulder_wrist_angle"] = exp_smooth(smooth_store["right_shoulder_wrist_angle"], r_sw_angle, SMOOTH_ALPHA)
                smooth_store["right_hor_ratio"] = exp_smooth(smooth_store["right_hor_ratio"], r_hor_ratio, SMOOTH_ALPHA)

                right_state = classify_arm(
                    smooth_store["right_elbow_angle"],
                    smooth_store["right_shoulder_wrist_angle"],
                    smooth_store["right_hor_ratio"],
                    rsy, min(lhipy, rhipy)
                )

            guessed = decide_state(left_state, right_state)
            confirmed = hyst.update(guessed, True)
            if confirmed:
                current_state = guessed

        cv2.imshow("Camera (Video)", vis)

        # Intentar obtener imagen del robot (dinámicamente si era None)
        img = None
        if current_state:
            img = _load_image_for_state_dynamic(current_state)

        if img is not None:
            # Si tiene alpha, componer sobre fondo blanco
            try:
                if img.shape[2] == 4:
                    alpha = img[:, :, 3] / 255.0
                    bgr = img[:, :, :3]
                    bg = np.ones_like(bgr, dtype=np.uint8) * 255
                    img_disp = (bgr * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
                else:
                    img_disp = img
            except Exception:
                img_disp = img
            resized = cv2.resize(img_disp, (400, 400))
            cv2.imshow("Robot", resized)
        else:
            # mostrar placeholder con indicación amigable
            placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 220
            text = f"Sin imagen para:\n{current_state}"
            # dividir texto en dos líneas si necesario
            y0 = 160
            for i, line in enumerate(text.splitlines()):
                cv2.putText(placeholder, line, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.imshow("Robot", placeholder)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_ctx.close()




# Celda 8 (MODIFICADA) - seleccionar video con diálogo
if __name__ == "__main__":
    # Abre diálogo para seleccionar el archivo de video
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Selecciona un video con movimientos",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_path:
        print("No seleccionaste archivo. Ejecuta de nuevo y selecciona un video.")
    else:
        print("Procesando:", video_path)
        run(video_path)

