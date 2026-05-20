import os, cv2, zlib, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fftpack import dct, idct

FRAMES_DIR    = 'images/'
OUTPUT_DIR    = 'output/'
GOP_SIZE      = 10
SEARCH_WINDOW = 8
FQ            = 5
MB_SIZE       = 16
RESIZE_W      = 320
RESIZE_H      = 180

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger et redimensionner 320x180
files = sorted(Path(FRAMES_DIR).glob('*.jpg'))
frames_bgr = []
for f in files:
    img = cv2.imread(str(f))
    if img is not None:
        img = cv2.resize(img, (RESIZE_W, RESIZE_H))
        frames_bgr.append(img)

N_FRAMES = len(frames_bgr)

##1
def bgr_to_ycbcr(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Y  = ycrcb[:, :, 0].astype(np.float32)
    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)
    return Y, Cb, Cr

def ycbcr_to_bgr(Y, Cb, Cr):
    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Y, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

def chroma_subsample(Cb, Cr):
    return Cb[::2, ::2], Cr[::2, ::2]

def chroma_upsample(Cb_sub, Cr_sub, shape):
    H, W = shape
    Cb_up = cv2.resize(Cb_sub, (W, H), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (W, H), interpolation=cv2.INTER_LINEAR)
    return Cb_up, Cr_up

frames_pre = []
for frame in frames_bgr:
    Y, Cb, Cr  = bgr_to_ycbcr(frame)
    Cb_s, Cr_s = chroma_subsample(Cb, Cr)
    frames_pre.append({'Y': Y, 'Cb': Cb_s, 'Cr': Cr_s, 'shape': Y.shape})

print(f' Étape 1 : {len(frames_pre)} frames converties')
print(f'   Y  : {frames_pre[0]["Y"].shape}  ← pleine resolution')
print(f'   Cb : {frames_pre[0]["Cb"].shape} ← divise par 2 (4:2:0)')
print(f'   Cr : {frames_pre[0]["Cr"].shape} ← divise par 2 (4:2:0)')

##2

# Matrice de quantisation : Q(i,j) = 1 + (1+i+j) * Fq  (formule cours)
def make_quant_matrix(fq, n=8):
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return (1 + (1 + i_idx + j_idx) * fq).astype(np.float32)

Q_MATRIX = make_quant_matrix(FQ)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def encode_channel_dct(channel, Q):
    H, W  = channel.shape
    H_pad = (H + 7) // 8 * 8
    W_pad = (W + 7) // 8 * 8
    padded = np.zeros((H_pad, W_pad), dtype=np.float32)
    padded[:H, :W] = channel - 128
    coeffs = np.zeros_like(padded)
    for i in range(0, H_pad, 8):
        for j in range(0, W_pad, 8):
            coeffs[i:i+8, j:j+8] = np.round(dct2(padded[i:i+8, j:j+8]) / Q)
    return coeffs[:H, :W].astype(np.int16)

def decode_channel_dct(coeffs, Q, shape):
    H, W  = shape
    H_pad = (H + 7) // 8 * 8
    W_pad = (W + 7) // 8 * 8
    padded = np.zeros((H_pad, W_pad), dtype=np.float32)
    padded[:H, :W] = coeffs.astype(np.float32)
    recon = np.zeros_like(padded)
    for i in range(0, H_pad, 8):
        for j in range(0, W_pad, 8):
            recon[i:i+8, j:j+8] = idct2(padded[i:i+8, j:j+8] * Q)
    return np.clip(recon[:H, :W] + 128, 0, 255).astype(np.float32)

def encode_iframe(frame_pre):
    return {
        'type' : 'I',
        'Y'    : encode_channel_dct(frame_pre['Y'],  Q_MATRIX),
        'Cb'   : encode_channel_dct(frame_pre['Cb'], Q_MATRIX),
        'Cr'   : encode_channel_dct(frame_pre['Cr'], Q_MATRIX),
        'shape': frame_pre['shape']
    }

def decode_iframe(enc):
    H, W = enc['shape']
    Y    = decode_channel_dct(enc['Y'],  Q_MATRIX, (H,     W    ))
    Cb   = decode_channel_dct(enc['Cb'], Q_MATRIX, (H//2,  W//2 ))
    Cr   = decode_channel_dct(enc['Cr'], Q_MATRIX, (H//2,  W//2 ))
    Cb_up, Cr_up = chroma_upsample(Cb, Cr, (H, W))
    return ycbcr_to_bgr(Y, Cb_up, Cr_up)

print(f' Étape 2 : fonctions DCT definies  |  Q_MATRIX (Fq={FQ}) :')
print(Q_MATRIX.astype(int))

##3

def block_matching(cur_Y, ref_Y):
    H, W = cur_Y.shape
    mvs  = []
    for i in range(0, H - MB_SIZE + 1, MB_SIZE):
        for j in range(0, W - MB_SIZE + 1, MB_SIZE):
            cur_block = cur_Y[i:i+MB_SIZE, j:j+MB_SIZE].astype(np.float32)
            best_mse  = float('inf')
            best_dx, best_dy = 0, 0
            dy_min = max(-SEARCH_WINDOW, -i)
            dy_max = min(SEARCH_WINDOW,  H - MB_SIZE - i)
            dx_min = max(-SEARCH_WINDOW, -j)
            dx_max = min(SEARCH_WINDOW,  W - MB_SIZE - j)
            for dy in range(dy_min, dy_max + 1):
                for dx in range(dx_min, dx_max + 1):
                    ref_block = ref_Y[i+dy:i+dy+MB_SIZE, j+dx:j+dx+MB_SIZE].astype(np.float32)
                    mse = np.mean((cur_block - ref_block) ** 2)
                    if mse < best_mse:
                        best_mse  = mse
                        best_dx, best_dy = dx, dy
            mvs.append((i, j, best_dx, best_dy))
    return mvs

def encode_pframe(frame_pre, ref_bgr):
    H, W      = frame_pre['shape']
    cur_Y     = frame_pre['Y']
    ref_Y, _, _ = bgr_to_ycbcr(ref_bgr)
    mvs       = block_matching(cur_Y, ref_Y)
    residual  = np.zeros((H, W), dtype=np.float32)
    for (i, j, dx, dy) in mvs:
        ri = np.clip(i + dy, 0, H - MB_SIZE)
        rj = np.clip(j + dx, 0, W - MB_SIZE)
        residual[i:i+MB_SIZE, j:j+MB_SIZE] = (
            cur_Y[i:i+MB_SIZE, j:j+MB_SIZE] -
            ref_Y[ri:ri+MB_SIZE, rj:rj+MB_SIZE]
        )
    return {
        'type' : 'P',
        'mvs'  : mvs,
        'res_Y': encode_channel_dct(residual + 128, Q_MATRIX),
        'Cb'   : encode_channel_dct(frame_pre['Cb'], Q_MATRIX),
        'Cr'   : encode_channel_dct(frame_pre['Cr'], Q_MATRIX),
        'shape': frame_pre['shape']
    }

def decode_pframe(enc, ref_bgr):
    H, W        = enc['shape']
    ref_Y, _, _ = bgr_to_ycbcr(ref_bgr)
    res_Y       = decode_channel_dct(enc['res_Y'], Q_MATRIX, (H, W)) - 128
    recon_Y     = np.zeros((H, W), dtype=np.float32)
    for (i, j, dx, dy) in enc['mvs']:
        ri = np.clip(i + dy, 0, H - MB_SIZE)
        rj = np.clip(j + dx, 0, W - MB_SIZE)
        recon_Y[i:i+MB_SIZE, j:j+MB_SIZE] = (
            ref_Y[ri:ri+MB_SIZE, rj:rj+MB_SIZE] +
            res_Y[i:i+MB_SIZE,   j:j+MB_SIZE]
        )
    recon_Y      = np.clip(recon_Y, 0, 255)
    Cb           = decode_channel_dct(enc['Cb'], Q_MATRIX, (H//2, W//2))
    Cr           = decode_channel_dct(enc['Cr'], Q_MATRIX, (H//2, W//2))
    Cb_up, Cr_up = chroma_upsample(Cb, Cr, (H, W))
    return ycbcr_to_bgr(recon_Y, Cb_up, Cr_up)

print(' Étape 3 : fonctions P-frames definies')

##
encoded_frames = []
reconstructed  = []
frame_types    = []

for idx in range(N_FRAMES):
    if idx % GOP_SIZE == 0:
        enc = encode_iframe(frames_pre[idx])
        dec = decode_iframe(enc)
        frame_types.append('I')
        print(f'  Frame {idx:03d} → I-frame')
    else:
        enc = encode_pframe(frames_pre[idx], reconstructed[-1])
        dec = decode_pframe(enc, reconstructed[-1])
        frame_types.append('P')
        print(f'  Frame {idx:03d} → P-frame')
    encoded_frames.append(enc)
    reconstructed.append(dec)

print(f'\n Encodage termine : {frame_types.count("I")} I-frames + {frame_types.count("P")} P-frames')

##4
BIN_PATH = OUTPUT_DIR + 'video_compressed.bin'

def encode_to_bin(encoded_frames, path):
    raw        = pickle.dumps(encoded_frames)
    compressed = zlib.compress(raw, level=9)
    with open(path, 'wb') as f:
        f.write(compressed)
    return len(raw), len(compressed)

def decode_from_bin(path):
    with open(path, 'rb') as f:
        compressed = f.read()
    return pickle.loads(zlib.decompress(compressed))

raw_size, bin_size = encode_to_bin(encoded_frames, BIN_PATH)
H0, W0 = frames_pre[0]['shape']
original_size     = N_FRAMES * H0 * W0 * 3
compression_ratio = original_size / bin_size
decoded_frames    = decode_from_bin(BIN_PATH)

print(f' Étape 4')
print(f'   Taille originale  : {original_size/1024:.0f} KB')
print(f'   Taille compressee : {bin_size/1024:.0f} KB')
print(f'   Ratio compression : {compression_ratio:.2f}x')
print(f'   Decodage verifie  : {len(decoded_frames)} frames')

##5

fig, axes = plt.subplots(4, 5, figsize=(20, 14))
fig.suptitle('Pipeline MPEG-4 — Visualisation complete', fontsize=14, fontweight='bold')

#  Frames originales
for k in range(5):
    ax = axes[0, k]
    if k < N_FRAMES:
        ax.imshow(cv2.cvtColor(frames_bgr[k], cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {k} ({frame_types[k]})', fontsize=8)
    ax.axis('off')

#  Canaux Y, Cb, Cr
Y0, Cb0, Cr0 = bgr_to_ycbcr(frames_bgr[0])
axes[1, 0].imshow(Y0,  cmap='gray');  axes[1, 0].set_title('Y — Luminance',    fontsize=8); axes[1, 0].axis('off')
axes[1, 1].imshow(Cb0, cmap='Blues'); axes[1, 1].set_title('Cb — Chroma bleu', fontsize=8); axes[1, 1].axis('off')
axes[1, 2].imshow(Cr0, cmap='Reds');  axes[1, 2].set_title('Cr — Chroma rouge',fontsize=8); axes[1, 2].axis('off')
axes[1, 3].axis('off')
axes[1, 4].axis('off')

# Bloc 8x8 : raw → DCT → quantisé → reconstruit 
block_raw   = Y0[16:24, 16:24].copy()
block_dct   = dct2(block_raw - 128)
block_quant = np.round(block_dct / Q_MATRIX)
block_recon = np.clip(idct2(block_quant * Q_MATRIX) + 128, 0, 255)

axes[2, 0].imshow(block_raw,   cmap='gray');   axes[2, 0].set_title('Pixels bruts 8x8', fontsize=8); axes[2, 0].axis('off')
axes[2, 1].imshow(block_dct,   cmap='RdBu_r'); axes[2, 1].set_title('DCT',              fontsize=8); axes[2, 1].axis('off')
axes[2, 2].imshow(block_quant, cmap='RdBu_r'); axes[2, 2].set_title('Quantisé',         fontsize=8); axes[2, 2].axis('off')
axes[2, 3].imshow(block_recon, cmap='gray');   axes[2, 3].set_title('Reconstruit',       fontsize=8); axes[2, 3].axis('off')
axes[2, 4].axis('off')

#Vecteurs de mouvement + Résidus + Reconstruction
H0, W0 = frames_pre[0]['shape']
p_idx  = next((i for i, t in enumerate(frame_types) if t == 'P'), None)

ax = axes[3, 0]
if p_idx is not None:
    ax.imshow(cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB))
    for (i, j, dx, dy) in encoded_frames[p_idx]['mvs']:
        if abs(dx) > 0 or abs(dy) > 0:
            ax.annotate('', xy=(j+dx+MB_SIZE//2, i+dy+MB_SIZE//2),
                        xytext=(j+MB_SIZE//2, i+MB_SIZE//2),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
        else:
            ax.plot(j+MB_SIZE//2, i+MB_SIZE//2, 'g.', markersize=3)
ax.set_title(f'Vecteurs mouvement\nP-frame {p_idx}', fontsize=8)
ax.axis('off')

ax = axes[3, 1]
if p_idx is not None:
    res = decode_channel_dct(encoded_frames[p_idx]['res_Y'], Q_MATRIX, (H0, W0)) - 128
    im  = ax.imshow(res, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title('Résidu Y (P-frame)', fontsize=8)
ax.axis('off')

ax = axes[3, 2]
ax.imshow(cv2.cvtColor(reconstructed[0], cv2.COLOR_BGR2RGB))
ax.set_title('Reconstruit 0 (I)', fontsize=8)
ax.axis('off')

ax = axes[3, 3]
if p_idx is not None:
    ax.imshow(cv2.cvtColor(reconstructed[p_idx], cv2.COLOR_BGR2RGB))
    ax.set_title(f'Reconstruit {p_idx} (P)', fontsize=8)
ax.axis('off')

axes[3, 4].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_DIR + 'pipeline_visualisation.png', dpi=120, bbox_inches='tight')
plt.show()

##6 — Analyse expérimentale : Fq vs ratio & GOP vs ratio

def encode_video_params(fq, gop_size):
    """Ré-encode toute la vidéo avec un Fq et GOP donnés, retourne taille en KB."""
    Q = make_quant_matrix(fq)
    frames_pre_local = []
    for frame in frames_bgr:
        Y, Cb, Cr  = bgr_to_ycbcr(frame)
        Cb_s, Cr_s = chroma_subsample(Cb, Cr)
        frames_pre_local.append({'Y': Y, 'Cb': Cb_s, 'Cr': Cr_s, 'shape': Y.shape})

    encoded_local = []
    recon_local   = []

    for idx in range(N_FRAMES):
        H, W = frames_pre_local[idx]['shape']
        if idx % gop_size == 0:
            enc = {
                'type' : 'I',
                'Y'    : encode_channel_dct(frames_pre_local[idx]['Y'],  Q),
                'Cb'   : encode_channel_dct(frames_pre_local[idx]['Cb'], Q),
                'Cr'   : encode_channel_dct(frames_pre_local[idx]['Cr'], Q),
                'shape': (H, W)
            }
            Yd  = decode_channel_dct(enc['Y'],  Q, (H,   W  ))
            Cbd = decode_channel_dct(enc['Cb'], Q, (H//2,W//2))
            Crd = decode_channel_dct(enc['Cr'], Q, (H//2,W//2))
            Cbu, Cru = chroma_upsample(Cbd, Crd, (H, W))
            dec = ycbcr_to_bgr(Yd, Cbu, Cru)
        else:
            ref_Y, _, _ = bgr_to_ycbcr(recon_local[-1])
            mvs      = block_matching(frames_pre_local[idx]['Y'], ref_Y)
            residual = np.zeros((H, W), dtype=np.float32)
            for (i, j, dx, dy) in mvs:
                ri = np.clip(i+dy, 0, H-MB_SIZE)
                rj = np.clip(j+dx, 0, W-MB_SIZE)
                residual[i:i+MB_SIZE, j:j+MB_SIZE] = (
                    frames_pre_local[idx]['Y'][i:i+MB_SIZE, j:j+MB_SIZE]
                    - ref_Y[ri:ri+MB_SIZE, rj:rj+MB_SIZE])
            enc = {
                'type' : 'P',
                'mvs'  : mvs,
                'res_Y': encode_channel_dct(residual + 128, Q),
                'Cb'   : encode_channel_dct(frames_pre_local[idx]['Cb'], Q),
                'Cr'   : encode_channel_dct(frames_pre_local[idx]['Cr'], Q),
                'shape': (H, W)
            }
            res_dec  = decode_channel_dct(enc['res_Y'], Q, (H, W)) - 128
            recon_Y  = np.zeros((H, W), dtype=np.float32)
            for (i, j, dx, dy) in mvs:
                ri = np.clip(i+dy, 0, H-MB_SIZE)
                rj = np.clip(j+dx, 0, W-MB_SIZE)
                recon_Y[i:i+MB_SIZE, j:j+MB_SIZE] = (
                    ref_Y[ri:ri+MB_SIZE, rj:rj+MB_SIZE]
                    + res_dec[i:i+MB_SIZE, j:j+MB_SIZE])
            recon_Y  = np.clip(recon_Y, 0, 255)
            Cbd = decode_channel_dct(enc['Cb'], Q, (H//2, W//2))
            Crd = decode_channel_dct(enc['Cr'], Q, (H//2, W//2))
            Cbu, Cru = chroma_upsample(Cbd, Crd, (H, W))
            dec = ycbcr_to_bgr(recon_Y, Cbu, Cru)

        encoded_local.append(enc)
        recon_local.append(dec)

    compressed = zlib.compress(pickle.dumps(encoded_local), level=9)
    return len(compressed) / 1024


original_kb = N_FRAMES * RESIZE_H * RESIZE_W * 3 / 1024

# ── Graphique 1 : Fq vs ratio de compression ──────────────────
FQ_VALUES = [1, 2, 3, 5, 8, 10, 15, 20]
print('\n Étape 6a : variation de Fq (GOP fixé à 10)...')
ratios_fq = []
for fq in FQ_VALUES:
    kb    = encode_video_params(fq, GOP_SIZE)
    ratio = original_kb / kb
    ratios_fq.append(ratio)
    print(f'   Fq={fq:2d} → {kb:.0f} KB  ratio={ratio:.2f}x')

fig_fq, ax_fq = plt.subplots(figsize=(8, 5))
ax_fq.plot(FQ_VALUES, ratios_fq, 'o-', color='steelblue',
           linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax_fq.axvline(x=FQ, color='red', linestyle='--', linewidth=1.5,
              label=f'Fq utilisé dans le projet (Fq={FQ})')
ax_fq.set_xlabel('Facteur de quantification Fq', fontsize=12)
ax_fq.set_ylabel('Taux de compression (×)', fontsize=12)
ax_fq.set_title('Taux de compression vs Facteur de quantification\n(GOP=10, 30 frames 320×180)', fontsize=13)
ax_fq.legend(fontsize=10)
ax_fq.grid(True, linestyle='--', alpha=0.5)
ax_fq.set_xticks(FQ_VALUES)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'graph_fq_vs_ratio.png', dpi=120)
plt.show()
print(' → Sauvegardé : output/graph_fq_vs_ratio.png')

# ── Graphique 2 : GOP vs ratio de compression ─────────────────
GOP_VALUES = [1, 2, 5, 10, 15, 30]
print('\n Étape 6b : variation du GOP (Fq fixé à 5)...')
ratios_gop = []
for gop in GOP_VALUES:
    kb    = encode_video_params(FQ, gop)
    ratio = original_kb / kb
    ratios_gop.append(ratio)
    print(f'   GOP={gop:2d} → {kb:.0f} KB  ratio={ratio:.2f}x')

fig_gop, ax_gop = plt.subplots(figsize=(8, 5))
ax_gop.plot(GOP_VALUES, ratios_gop, 's-', color='darkorange',
            linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax_gop.axvline(x=GOP_SIZE, color='red', linestyle='--', linewidth=1.5,
               label=f'GOP utilisé dans le projet (GOP={GOP_SIZE})')
ax_gop.set_xlabel('Taille du GOP', fontsize=12)
ax_gop.set_ylabel('Taux de compression (×)', fontsize=12)
ax_gop.set_title('Taux de compression vs Taille du GOP\n(Fq=5, 30 frames 320×180)', fontsize=13)
ax_gop.legend(fontsize=10)
ax_gop.grid(True, linestyle='--', alpha=0.5)
ax_gop.set_xticks(GOP_VALUES)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'graph_gop_vs_ratio.png', dpi=120)
plt.show()
print(' → Sauvegardé : output/graph_gop_vs_ratio.png')

