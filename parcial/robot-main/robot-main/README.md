# Simulación de movimientos de brazos con MediaPipe
Repositorio / Trabajo práctico — UNAB — Ciencia de Datos  
Duración propuesta: **40 minutos**

---

## Descripción breve

![](https://raw.githubusercontent.com/adiacla/robot/refs/heads/main/salida.png)


Los estudiantes desarrollarán una aplicación en Python llamada `app.py` (o `robot.py`) que utiliza MediaPipe para detectar los landmarks de la pose humana, calcula ángulos y relaciones relevantes de los brazos y muestra simultáneamente:

- Ventana izquierda: tu cámara / video con landmarks y overlays de debug.
- Ventana derecha: pantalla del "robot" que muestra imágenes según el estado detectado (ambos abajo, ambos arriba, brazo izquierdo extendido, etc.).

La app debe funcionar con cámara en vivo (por defecto) o con un archivo de vídeo pasado por parámetro.

---

## Objetivos educativos
- Introducción práctica a MediaPipe como modelo de IA para detección de pose.
- Procesamiento de landmarks: convertir coordenadas normalizadas a píxeles.
- Cálculo de ángulos (codo, hombro→muñeca) y ratios (componentes normalizadas).
- Diseño de reglas heurísticas para estados (extended / up / down) y manejo de ruido (suavizado + histeresis).
- Implementación de interfaz visual (OpenCV/pygame) y control básico (teclas).

---

## Requisitos y versión recomendada
- Python 3.10 (o 3.11)
- Sistema con cámara o vídeo de prueba
- Recomendado: entorno virtual (venv) o conda

requirements.txt recomendado:
```text
numpy>=1.23
opencv-python>=4.6
pygame>=2.1
mediapipe>=0.10
```

Nota: en algunas plataformas (macOS Apple Silicon) la instalación de `mediapipe` puede necesitar conda o ruedas compatibles. Si aparece un warning de protobuf como:
```
Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.
```
actualiza `protobuf`:
```bash
pip install --upgrade protobuf
```
o aplica el shim temporal incluido en el repo.

---

## Archivos esperados en la entrega
- `app.py` o `robot.py` — script principal ejecutable
- `robot/` — carpeta con imágenes (nombres exactos):
  - `ambos_abajo.png`
  - `ambos_arriba.png`
  - `ambos_extendida.png`
  - `derecha_arriba_izq_abajo.png`
  - `derecha_arriba_izq_extendida.png`
  - `derecha_extendida_izq_abajo.png`
  - `izquierda_arriba_der_abajo.png`
  - `izquierda_arriba_der_extendida.png`
  - `izquierda_extendida_der_abajo.png`
- `requirements.txt`
- `README.md` (este archivo)
- `answers.txt` — respuestas a las 5 preguntas requeridas

---

## Comandos para configurar y ejecutar (rápido)
1. Crear y activar entorno virtual (ejemplo Unix/macOS):
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Ejecutar (usar cámara por defecto):
```bash
python app.py
```

3. Ejecutar con vídeo (si cámara no disponible):
```bash
python app.py --video path/to/video.mp4
```

## Requerimientos mínimos funcionales (lo que se debe implementar en 40 minutos)

![](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F34clr3wvqxhjk35nl45p.png)

![](https://raw.githubusercontent.com/adiacla/robot/refs/heads/main/calculos.png)

1. Captura de vídeo desde cámara o carga de archivo.
2. Uso de MediaPipe (Pose/Holistic) para obtener `results.pose_landmarks`.
3. Conversión de landmarks a píxeles: `px = int(landmark.x * width)`, `py = int(landmark.y * height)`.
4. Cálculos por brazo:
   - Ángulo de codo: ángulo en el punto codo entre hombro–codo–muñeca.
   - Dirección hombro→muñeca: atan2 para saber si brazo apunta hacia arriba/abajo/horizontal.
   - Ratio horizontal normalizada: componente horizontal (|wx - sx|) / longitud_total_brazo.
5. Detección de estados (mínimo):
   - `both_down`, `both_up`, `both_extended`,
   - `right_up_left_down`, `right_up_left_extended`,
   - `right_extended_left_down`, `left_up_right_down`, `left_up_right_extended`, `left_extended_right_down`
6. Visualización:
   - Ventana con la cámara/video con landmarks y overlay (cv2).
   - Ventana que muestra la imagen del robot según el estado (puede ser Pygame o cv2).
7. Mecanismo anti-ruido:
   - Suavizado exponencial o median filter para ángulos/ratios.
   - Histeresis por conteo de frames (ej.: CONFIRM_FRAMES = 3-5).



---

## Indicaciones técnicas rápidas
- Inicializar MediaPipe:
```python
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    results = holistic.process(image_rgb)
```
- Función ángulo de codo:
```python
def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
```
- Ángulo hombro→muñeca:
```python
vx = wrist_x - shoulder_x
vy = shoulder_y - wrist_y  # invertir Y en coords píxel
angle = math.degrees(math.atan2(vy, vx))  # 0 = derecha, 90 = arriba, -90 = abajo
```
- Ratio horizontal normalizada:
```python
arm_len = dist(shoulder, elbow) + dist(elbow, wrist)
hor_ratio = abs(wrist_x - shoulder_x) / max(1.0, arm_len)
```
- Heurística ejemplo:
  - Extended: codo > 150° y hor_ratio > 0.55
  - Up: wrist_y < shoulder_y
  - Down: wrist_y > hip_y o shoulder→wrist angle < -60°
  - Confirmar estados con `CONFIRM_FRAMES = 3` (incrementar contador por frame si la condición se cumple, decrementar si no).

---

## Rubrica de evaluación (100 pts)
- 45 pts — Funcionalidad:
  - 20 pts: correcta extracción de landmarks y cálculo de ángulos.
  - 10 pts: lógica de estados y mapping a imágenes.
  - 10 pts: entrada cámara/video y funcionamiento estable.
  - 5 pts: manejo básico de ruido (suavizado/histeresis).
- 25 pts — Visual/UX:
  - 10 pts: muestra ambas vistas correctamente.
  - 10 pts: overlays legibles con valores clave.
  - 5 pts: transiciones visuales aceptables entre imágenes.
- 20 pts — Código y documentación:
  - 10 pts: código legible, modular y comentado.
  - 10 pts: README + answers.txt con instrucciones y respuestas.
- 10 pts — Respuestas teóricas:
  - Respuestas claras y correctas a las 5 preguntas (siguientes sección).

Criterio mínimo para aprobar la tarea: app ejecutable con las dos vistas y `both_extended`, `both_up`, `both_down` funcionales en pruebas básicas.

---

## Preguntas (responder en `answers.txt`)
1. Breve (3–6 líneas): ¿Cómo detecta MediaPipe los landmarks corporales? Describe qué devuelve y qué tipo de confianza/visibilidad trae.
2. Matemáticamente: ¿Cómo calculaste el ángulo de codo y por qué es correcto usar productos punto / arccos para ello?
3. ¿Qué es la "ratio horizontal normalizada" y por qué ayuda a distinguir brazo extendido de brazo pegado al cuerpo?
4. ¿Qué técnicas implementaste para reducir ruido y parpadeos (explica smoothing y/o histeresis) y por qué funcionan?
5. Menciona dos limitaciones del enfoque (ej.: oclusión, iluminación, ángulo de cámara) y propone una mejora concreta para cada una.

---

## Checklist de entrega (para el estudiante)
- [ ] `app.py` / `robot.py` funcionando
- [ ] `robot/` con imágenes con nombres correctos
- [ ] `requirements.txt`
- [ ] `README.md` (este archivo)

---

## Evaluación práctica / recomendaciones para el docente
- Dar un starter kit si se desea (sugerido): `app_base.py` con:
  - inicialización de cámara/MediaPipe,
  - función `angle_between`,
  - template minimal para dos vistas (sin la lógica completa).
- Tiempo total recomendado: 40 minutos. Ajustar la consigna si no se proporciona starter kit.
- Para la corrección: pedir que el alumno grabe 30–60s de demo o capture 4 capturas (both_up, both_extended, both_down, left_up_right_down).

---

## Extensiones (bonus ideas)
- Calibración interactiva: presión de tecla para capturar ejemplos (`h` para brazo horizontal, `d` para brazo abajo) y ajustar automáticamente thresholds.
- Dibujar robot vectorial en Pygame y rotar brazos por ángulos reales en vez de imágenes fijas.
- Añadir control por teclado para cambiar duración de transición y CONFIRM_FRAMES en tiempo real.
- Guardar logs de detección y métricas de rendimiento (FPS).

---

## Referencias útiles
- MediaPipe Pose / Holistic documentation: https://developers.google.com/mediapipe
- OpenCV Python tutorials: https://opencv.org
- Pygame docs: https://www.pygame.org/docs/

