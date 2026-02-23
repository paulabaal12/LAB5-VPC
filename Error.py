import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img_left = cv2.imread('img/izquierda1.jpeg')
img_right = cv2.imread('img/derecha1.jpeg')

# escala de grises
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# SIFT y matching FLANN
sift = cv2.SIFT_create()
kp_left, des_left = sift.detectAndCompute(gray_left, None)  # Train (destino)
kp_right, des_right = sift.detectAndCompute(gray_right, None)  # Query (origen)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Derecha es query, izquierda es train
matches = flann.knnMatch(des_right, des_left, k=2)
# print(matches)
# print(len(matches))

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Good matches encontrados: {len(good_matches)}")

img_matches = cv2.drawMatches(img_right, kp_right, img_left, kp_left, good_matches, None, flags=2)


# Homografía

# Extraer coordenadas de los x, y
src_pts = np.float32([kp_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp * deformación y cálculo del canvas
h_left, w_left = img_left.shape[:2]
h_right, w_right = img_right.shape[:2]


#Ancho de  la imagen 1 + el ancho de la imagen 2  (Ancho = i1.w + i2.w)
# alto = de la imagen de imagen más alto
canvas_w = w_left + w_right
canvas_h = max(h_left, h_right)


panorama = cv2.warpPerspective(img_right, M, (canvas_w, canvas_h))
panorama[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

dst_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
plt.imshow(dst_rgb)
plt.show()

# --- 1) Warp de la imagen derecha al canvas ---
h_left, w_left = img_left.shape[:2]
h_right, w_right = img_right.shape[:2]

canvas_w = w_left + w_right
canvas_h = max(h_left, h_right)

warped_right = cv2.warpPerspective(img_right, M, (canvas_w, canvas_h))

# Poner la izquierda en el canvas también
left_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
left_canvas[0:h_left, 0:w_left] = img_left

# --- 2) Crear máscaras de contenido (para no mezclar zonas negras vacías) ---
mask_warp = cv2.warpPerspective(np.ones((h_right, w_right), dtype=np.uint8)*255, M, (canvas_w, canvas_h))
mask_left = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
mask_left[0:h_left, 0:w_left] = 255

# --- 3) Convertir a gris y construir anaglifo: R=base, G=B=warped ---
gray_left = cv2.cvtColor(left_canvas, cv2.COLOR_BGR2GRAY)
gray_warp = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)

# Normalizar a 0..255 (uint8) por seguridad
gray_left_u8 = cv2.normalize(gray_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gray_warp_u8 = cv2.normalize(gray_warp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

anaglyph = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Canal Rojo = izquierda (solo donde hay izquierda)
anaglyph[..., 0] = 0  # (esto sería Azul en BGR, lo corregimos abajo con RGB)
anaglyph[..., 1] = 0
anaglyph[..., 2] = 0

# Construimos en RGB directamente para mostrar con matplotlib
anaglyph_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# R
anaglyph_rgb[..., 0] = gray_left_u8 * (mask_left > 0)

# G y B (cian)
anaglyph_rgb[..., 1] = gray_warp_u8 * (mask_warp > 0)
anaglyph_rgb[..., 2] = gray_warp_u8 * (mask_warp > 0)



import numpy as np
import matplotlib.pyplot as plt

# Si te va lento o no registra clicks en notebook, prueba antes:
# %matplotlib qt

plt.figure(figsize=(16, 9))
plt.imshow(anaglyph_rgb)
plt.title(
    "TASK 2.2: Harás 6 clicks:\n"
    "1-2: A (Cercano) rojo->cian | 3-4: B (Medio) rojo->cian | 5-6: C (Fondo) rojo->cian\n"
    "Usa zoom para precisión."
)
plt.axis("off")
pts = plt.ginput(6, timeout=0)
plt.close()

def disparidad_x(p_rojo, p_cian):
    return abs(p_rojo[0] - p_cian[0])

A1, A2, B1, B2, C1, C2 = pts

filas = [
    ("A (Cercano)", A1, A2, disparidad_x(A1, A2)),
    ("B (Medio)",   B1, B2, disparidad_x(B1, B2)),
    ("C (Fondo)",   C1, C2, disparidad_x(C1, C2)),
]

print("{:<12} {:>9} {:>9} {:>9} {:>9} {:>14}".format("Objeto","x1","y1","x2","y2","|x1-x2| (px)"))
print("-"*70)
for nombre, p1, p2, d in filas:
    print("{:<12} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f} {:>14.2f}".format(
        nombre, p1[0], p1[1], p2[0], p2[1], d
    ))