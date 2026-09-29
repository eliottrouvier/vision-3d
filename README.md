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

- **`tracker_3d.py` / `one_euro_filter.py`** : Inférence MediaPipe Pose Landmarker + filtre adaptatif 1€ (suppression des micro-tremblements, ancrage au sol) et maillage filaire.
- **`gl_avatar_widget.py`** : Viewport PyOpenGL avec maillage continu facetté (SMPL-lite), alternance surface lisse / arêtes polygonales et éclairage studio.
- **Contrôles caméra 360° & Zoom** (dans `gl_avatar_widget.py`) : Orbite 3D (clic-glisser), zoom multi-modal (boutons tactiles `+`/`-`, pincement trackpad Mac, touches clavier `+`/`-` et molette), double-clic reset (`0`).
- **`main_window.py`** : Interface PySide6 avec vue scindée réglable, bascule Picture-in-Picture (PiP), scrubber temporel et contrôle de vitesse (0.5x à 2.0x).
- **`vision3d_launcher.sh` / `main.py`** : Point d'entrée CLI avec gestion automatique des chemins de plugins Qt QPA sous macOS.
