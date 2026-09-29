# vision-3d

Reconstruction 3D du corps humain (Human Mesh Recovery) et avatar articulé en temps réel (OpenGL) à partir d'une vidéo ou de la webcam.

## Lancement

```bash
# Lancement standard (échantillon kinématique)
vision3d

# Mode webcam directe (FaceTime HD / AVFoundation)
vision3d --webcam

# Vidéo personnalisée
vision3d chemin/vers/video.mp4
```

## Architecture

- **`tracker_3d.py`** : Inférence MediaPipe Pose Landmarker (coordonnées métriques $X, Y, Z$ en temps réel, ~18 ms sur Apple M4) et génération du maillage volumétrique filaire.
- **`gl_avatar_widget.py`** : Viewport PyOpenGL avec rendu du mannequin 3D ombré, grille de référence et double éclairage studio.
- **Contrôles caméra 360°** (dans `gl_avatar_widget.py`) : Orbite 3D à la souris (clic gauche : rotation, clic droit : pan, molette : zoom, double-clic : reset).
- **`main_window.py`** : Interface PySide6 avec vue scindée réglable, bascule Picture-in-Picture (PiP), scrubber temporel et contrôle de vitesse (0.5x à 2.0x).
- **`vision3d_launcher.sh` / `main.py`** : Point d'entrée CLI avec gestion automatique des chemins de plugins Qt QPA sous macOS.
