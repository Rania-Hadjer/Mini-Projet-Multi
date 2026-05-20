# Mini-Projet-Multi
Simplified MPEG-4 Video Encoder Pipeline
Mini projet Multimedia Systems — MPEG-4 Encoder Pipeline en Python.
Description
Ce projet implémente une version simplifiée d’un encodeur MPEG-4 en Python.
Le pipeline réalise :
1.	Conversion BGR → YCbCr
2.	Chroma subsampling 4:2:0
3.	Compression intra-image (I-frames) avec DCT
4.	Compression inter-image (P-frames) avec estimation de mouvement
5.	Quantification
6.	Compression binaire avec zlib
7.	Reconstruction des frames
8.	Visualisation complète du pipeline
9.	Analyse expérimentale :
o	Compression vs Quantization Factor
o	Compression vs GOP Size
Structure du projet
project/
│
├── encoder.py
├── images/
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   └── ...
│
├── output/
│
└── README.md
Installation
Installer les bibliothèques nécessaires :
pip install opencv-python numpy matplotlib scipy
Exécution
Lancer simplement :
python encoder.py
Paramètres principaux
GOP_SIZE      = 10
SEARCH_WINDOW = 8
FQ            = 5
MB_SIZE       = 16
RESIZE_W      = 320
RESIZE_H      = 180
Fonctionnement du pipeline
1. Prétraitement
•	Conversion BGR → YCbCr
•	Sous-échantillonnage chromatique 4:2:0
2. I-frames
Compression spatiale :
•	DCT 8×8
•	Quantification
•	IDCT
3. P-frames
Compression temporelle :
•	Découpage en macroblocs 16×16
•	Block Matching
•	Motion Vectors
•	Résidus
•	DCT + Quantification
4. Entropy Coding
Compression finale avec :
pickle + zlib

Fichiers générés
Le dossier output/ contient :
video_compressed.bin
pipeline_visualisation.png
graph_fq_vs_ratio.png
graph_gop_vs_ratio.png
Visualisations
Le programme génère automatiquement :
•	Frames originales
•	Canaux Y / Cb / Cr
•	DCT et quantification
•	Motion vectors
•	Résidus
•	Frames reconstruites
•	Graphes expérimentaux
Analyse expérimentale
Influence du facteur de quantification (FQ)
Le projet mesure l’effet de :
FQ = [1,2,3,5,8,10,15,20]
sur :
•	la taille compressée
•	le ratio de compression
Influence du GOP
Le projet mesure l’effet de :
GOP = [1,2,5,10,15,30]
sur le taux de compression.
Technologies utilisées
•	Python
•	OpenCV
•	NumPy
•	Matplotlib
•	SciPy
Auteur
Mini projet MPEG-4 — Multimedia Systems 2026

