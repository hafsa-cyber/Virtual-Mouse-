import cv2
import pyautogui
import numpy as np
import math
import time
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_styles


CAMERA_INDEX        = 0       
FRAME_WIDTH         = 800
FRAME_HEIGHT        = 600
SMOOTHING           = 3       
CLICK_DISTANCE      = 0.04   
DOUBLE_CLICK_DIST   = 0.10    
RIGHT_CLICK_DIST    = 0.04    
SCROLL_SENSITIVITY  = 20      
COOLDOWN            = 0.4     



hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)


screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


prev_x, prev_y     = 0, 0           
last_action_time   = 0              
prev_scroll_y      = None           
current_gesture    = "EN ATTENTE"  



def distance(p1, p2):
    """Calcule la distance euclidienne entre 2 landmarks normalisés."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def is_finger_up(landmarks, finger_tip, finger_pip):
    """
    Retourne True si le doigt est levé.
    Principe : le bout du doigt (tip) doit être AU-DESSUS de l'articulation (pip).
    En coordonnées image, Y est inversé → tip.y < pip.y signifie levé.
    """
    return landmarks[finger_tip].y < landmarks[finger_pip].y


def smooth_cursor(curr_x, curr_y, prev_x, prev_y, factor):
    """Lisse le mouvement du curseur pour éviter les tremblements."""
    smooth_x = prev_x + (curr_x - prev_x) / factor
    smooth_y = prev_y + (curr_y - prev_y) / factor
    return smooth_x, smooth_y


def can_act(cooldown=COOLDOWN):
    """Vérifie si le délai entre actions est respecté."""
    global last_action_time
    now = time.time()
    if now - last_action_time > cooldown:
        last_action_time = now
        return True
    return False


def detect_fingers(lm):

    fingers = {
        "pouce":        lm[4].x < lm[3].x,   # pouce levé si à gauche de son articulation
        "index":        is_finger_up(lm, 8,  6),
        "majeur":       is_finger_up(lm, 12, 10),
        "annulaire":    is_finger_up(lm, 16, 14),
        "auriculaire":  is_finger_up(lm, 20, 18),
    }
    return fingers


def detect_gesture(lm, fingers):
  

    # Distances clés
    dist_index_majeur   = distance(lm[8],  lm[12])   # scroll / double clic
    dist_pouce_index    = distance(lm[4],  lm[8])    # clic gauche / pincement
    dist_index_auricu   = distance(lm[8],  lm[20])   # clic droit

    # ── CLIC GAUCHE : pouce + index pincés
    if (dist_pouce_index < CLICK_DISTANCE
            and not fingers["majeur"]
            and not fingers["annulaire"]):
        return "CLIC GAUCHE"

    # ── CLIC DROIT : index + auriculaire levés, majeur baissé
    if (fingers["index"]
            and fingers["auriculaire"]
            and not fingers["majeur"]):
        return "CLIC DROIT"

    # ── SCROLL : index + majeur levés ET rapprochés
    if (fingers["index"]
            and fingers["majeur"]
            and not fingers["annulaire"]
            and dist_index_majeur < CLICK_DISTANCE):
        return "SCROLL"

    # ── DOUBLE CLIC : index + majeur levés ET écartés
    if (fingers["index"]
            and fingers["majeur"]
            and not fingers["annulaire"]
            and dist_index_majeur > DOUBLE_CLICK_DIST):
        return "DOUBLE CLIC"

    # ── DÉPLACEMENT : index seul levé
    if (fingers["index"]
            and not fingers["majeur"]
            and not fingers["annulaire"]
            and not fingers["auriculaire"]):
        return "DEPLACEMENT"

    return "EN ATTENTE"



GESTURE_COLORS = {
    "DEPLACEMENT":  (0,   255,  0),    # Vert
    "CLIC GAUCHE":  (255, 165,  0),    # Orange
    "DOUBLE CLIC":  (0,   200, 255),   # Cyan
    "CLIC DROIT":   (255,  50, 50),    # Rouge
    "SCROLL":       (180,  0,  255),   # Violet
    "EN ATTENTE":   (150, 150, 150),   # Gris
}

GESTURE_ICONS = {
    "DEPLACEMENT":  "[ INDEX SEUL ]",
    "CLIC GAUCHE":  "[ POUCE + INDEX pinces ]",
    "DOUBLE CLIC":  "[ INDEX + MAJEUR ecartes ]",
    "CLIC DROIT":   "[ INDEX + AURICULAIRE ]",
    "SCROLL":       "[ INDEX + MAJEUR proches ]",
    "EN ATTENTE":   "[ ... ]",
}


def draw_hud(frame, gesture, fingers, cursor_pos):
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture, (255, 255, 255))

    # Panneau semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Nom du geste
    cv2.putText(frame, f"GESTE: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Description des doigts
    icon = GESTURE_ICONS.get(gesture, "")
    cv2.putText(frame, icon, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Position curseur
    cx, cy = int(cursor_pos[0]), int(cursor_pos[1])
    cv2.putText(frame, f"Curseur: ({cx}, {cy})", (w - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Indicateurs doigts (bas de l'écran)
    finger_names  = ["POUCE", "INDEX", "MAJEUR", "ANNUL.", "AURICU."]
    finger_keys   = ["pouce", "index", "majeur", "annulaire", "auriculaire"]
    panel_y = h - 40
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 55), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    for i, (name, key) in enumerate(zip(finger_names, finger_keys)):
        x_pos = 10 + i * 125
        is_up = fingers.get(key, False)
        dot_color = (0, 255, 80) if is_up else (60, 60, 60)
        status    = "LEVE" if is_up else "BAISSE"
        cv2.circle(frame, (x_pos + 5, panel_y - 10), 6, dot_color, -1)
        cv2.putText(frame, name,   (x_pos + 16, panel_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, status, (x_pos + 5,  panel_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, dot_color, 1)

    # Cadre coloré selon geste
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 2)


# ══════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════

def main():
    global prev_x, prev_y, prev_scroll_y, current_gesture

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("=" * 50)
    print("  VIRTUAL MOUSE — DÉMARRÉ")
    print("=" * 50)
    print("  Appuie sur [Q] pour quitter\n")
    print("  GESTES :")
    print("  ✋ Déplacement  → Index levé SEUL")
    print("  🤏 Clic Gauche  → Pouce + Index pincés")
    print("  ✌️  Double Clic  → Index + Majeur écartés")
    print("  🤘 Clic Droit   → Index + Auriculaire levés")
    print("  👆 Clic Scroll  → Index + Majeur rapprochés (Haut/Bas écran)")
    print("=" * 50)

    cursor_pos = (screen_w // 2, screen_h // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Miroir horizontal (plus naturel)
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Conversion BGR → RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)

        fingers  = {k: False for k in ["pouce","index","majeur","annulaire","auriculaire"]}
        gesture  = "EN ATTENTE"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            # Dessiner les landmarks MediaPipe
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # ── Détection des doigts et geste
            fingers = detect_fingers(lm)
            gesture = detect_gesture(lm, fingers)
            current_gesture = gesture

            # ── Position de l'index (point 8) → coordonnées écran
            ix = lm[8].x  # normalisé 0..1
            iy = lm[8].y

            # Mapping caméra → écran avec marges (zone active = 20%..80%)
            margin = 0.2
            ix_mapped = np.interp(ix, [margin, 1 - margin], [0, screen_w])
            iy_mapped = np.interp(iy, [margin, 1 - margin], [0, screen_h])

            # Lissage
            sx, sy    = smooth_cursor(ix_mapped, iy_mapped, prev_x, prev_y, SMOOTHING)
            sx        = max(0, min(screen_w - 1, sx))
            sy        = max(0, min(screen_h - 1, sy))
            cursor_pos = (sx, sy)

            # ── Actions selon geste ──────────────────────────

            if gesture == "DEPLACEMENT":
                pyautogui.moveTo(sx, sy)
                prev_x, prev_y = sx, sy

            elif gesture == "CLIC GAUCHE":
                pyautogui.moveTo(sx, sy)
                if can_act():
                    pyautogui.click()
                    print("  🖱️  CLIC GAUCHE")
                prev_x, prev_y = sx, sy

            elif gesture == "DOUBLE CLIC":
                pyautogui.moveTo(sx, sy)
                if can_act(0.6):
                    pyautogui.doubleClick()
                    print("  🖱️  DOUBLE CLIC")
                prev_x, prev_y = sx, sy

            elif gesture == "CLIC DROIT":
                pyautogui.moveTo(sx, sy)
                if can_act():
                    pyautogui.rightClick()
                    print("  🖱️  CLIC DROIT")
                prev_x, prev_y = sx, sy

            elif gesture == "SCROLL":
                # Le scroll utilise la position Y de l'index pour déterminer le sens
                # Augmentation de la vitesse à 120 (un "cran" standard Windows)
                if iy < 0.4:  # Haut de l'écran
                    pyautogui.scroll(300)
                    print("  🖱️  SCROLL ↑ HAUT")
                elif iy > 0.6:  # Bas de l'écran
                    pyautogui.scroll(-300)
                    print("  🖱️  SCROLL ↓ BAS")
                
                # Optionnel : permettre aussi le mouvement pendant le scroll
                pyautogui.moveTo(sx, sy)
                prev_x, prev_y = sx, sy

            else:
                prev_x, prev_y = sx, sy

        # ── Affichage HUD
        draw_hud(frame, gesture, fingers, cursor_pos)

        cv2.imshow("Virtual Mouse — Hand Gesture AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Fermeture du programme.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()