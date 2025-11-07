import os
import re
import cv2
import glob
import argparse
import numpy as np
from typing import Dict, List, Tuple

# ============================================================
#  Assunzioni:
#   - (x, y) in unità "mondo", y positiva verso l'alto.
#   - La bounding box è centrata in (x, y), assi allineati al canvas.
#   - Se i tuoi dati hanno y positiva verso il basso, usa --no-invert-y.
# ============================================================

INVERT_Y = True  # y mondo verso l'alto -> inverti per il canvas OpenCV (y cresce verso il basso)

# -------------------------
# Utilità coordinate/canvas
# -------------------------

def world_to_screen(xs, ys, origin, scale):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    X = (xs - origin[0]) * scale
    Y = (ys - origin[1]) * scale
    if INVERT_Y:
        Y = -Y
    return X.astype(int), Y.astype(int)

def compute_bounds(data):
    x = data[:, 0]
    y = data[:, 1]
    w = data[:, 6]
    h = data[:, 7]
    xmin = np.min(x - w/2)
    xmax = np.max(x + w/2)
    ymin = np.min(y - h/2)
    ymax = np.max(y + h/2)
    return xmin, xmax, ymin, ymax

def make_canvas_params(data, scale=120, margin_px=60):
    xmin, xmax, ymin, ymax = compute_bounds(data)
    origin_x = xmin
    origin_y = (ymin if not INVERT_Y else ymax)
    width_world  = max(1e-6, (xmax - xmin))
    height_world = max(1e-6, (ymax - ymin))
    W = int(width_world  * scale) + 2 * margin_px
    H = int(height_world * scale) + 2 * margin_px
    return (origin_x, origin_y), (W, H), margin_px, scale

def draw_arrow(img, px, py, vx, vy, scale, color, thickness=2, tip=0.25):
    ex = int(px + vx * scale)
    ey = int(py + (-vy if INVERT_Y else vy) * scale)
    cv2.arrowedLine(img, (px, py), (ex, ey), color, thickness, tipLength=tip)

# --------------------------------
# Supporto multi-traiettoria (due serie per veicolo)
# --------------------------------

def compute_bounds_multi(series_dict):
    bounds = [compute_bounds(np.asarray(d, dtype=float)) for d in series_dict.values()]
    xmin = min(b[0] for b in bounds)
    xmax = max(b[1] for b in bounds)
    ymin = min(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)
    return xmin, xmax, ymin, ymax

def make_canvas_params_multi(series_dict, scale=120, margin_px=60):
    xmin, xmax, ymin, ymax = compute_bounds_multi(series_dict)
    origin_x = xmin
    origin_y = (ymin if not INVERT_Y else ymax)
    width_world  = max(1e-6, (xmax - xmin))
    height_world = max(1e-6, (ymax - ymin))
    W = int(width_world  * scale) + 2 * margin_px
    H = int(height_world * scale) + 2 * margin_px
    return (origin_x, origin_y), (W, H), margin_px, scale

def render_frame_multi(bg, series_dict, t, origin, margin, scale, styles,
                       show_trail=True, show_box=True, show_vectors=True, show_legend=True,
                       header_text=None):
    img = bg.copy()

    for name, data in series_dict.items():
        data = np.asarray(data, dtype=float)
        idx = min(t, len(data) - 1)

        if show_trail and idx >= 1:
            xs = data[:idx+1, 0]
            ys = data[:idx+1, 1]
            sx, sy = world_to_screen(xs, ys, origin, scale)
            sx += margin
            sy += margin
            pts = np.vstack([sx, sy]).T.reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=styles[name]['trail'],
                          thickness=2, lineType=cv2.LINE_AA)

        x, y, vx, vy, ax, ay, w, h = data[idx]
        px, py = world_to_screen(np.array([x]), np.array([y]), origin, scale)
        px = int(px[0]) + margin
        py = int(py[0]) + margin

        cv2.circle(img, (px, py), 4, styles[name]['trail'], -1, lineType=cv2.LINE_AA)

        if show_box:
            bw = int(max(1, w * scale))
            bh = int(max(1, h * scale))
            x1 = px - bw // 2
            y1 = py - bh // 2
            x2 = px + bw // 2
            y2 = py + bh // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), styles[name]['box'], 2)

        if show_vectors:
            draw_arrow(img, px, py, vx, vy, scale=20, color=styles[name]['vec'], thickness=2)
            draw_arrow(img, px, py, ax, ay, scale=60, color=styles[name].get('acc', styles[name]['vec']),
                       thickness=2)

    if header_text:
        cv2.putText(img, header_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)

    if show_legend:
        yleg = 54  # sotto all'header
        for name in series_dict.keys():
            col = styles[name]['trail']
            cv2.rectangle(img, (12, yleg - 10), (28, yleg + 6), col, -1)
            cv2.putText(img, name, (36, yleg + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1, cv2.LINE_AA)
            yleg += 22

    return img

# -------------------------
# Costruttori dati (dx,dy) o (x,y)
# -------------------------

def build_series_from_displacements(dxdy, dt=1.0, x0=0.0, y0=0.0, w=1.0, h=1.0):
    dxdy = np.asarray(dxdy, dtype=float)
    assert dxdy.ndim == 2 and dxdy.shape[1] >= 2, "CSV dxdy deve avere almeno due colonne (dx, dy)"
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    T = len(dx)

    x = np.empty(T, dtype=float)
    y = np.empty(T, dtype=float)
    x_prev, y_prev = x0, y0
    for t in range(T):
        x_curr = x_prev + dx[t]
        y_curr = y_prev + dy[t]
        x[t], y[t] = x_curr, y_curr
        x_prev, y_prev = x_curr, y_curr

    vx = dx / dt
    vy = dy / dt
    ax = np.zeros(T, dtype=float)
    ay = np.zeros(T, dtype=float)
    if T >= 2:
        ax[1:] = (vx[1:] - vx[:-1]) / dt
        ay[1:] = (vy[1:] - vy[:-1]) / dt

    w_arr = np.full(T, w, dtype=float)
    h_arr = np.full(T, h, dtype=float)
    return np.stack([x, y, vx, vy, ax, ay, w_arr, h_arr], axis=1)

def build_series_from_positions(xy, dt=1.0, w=1.0, h=1.0):
    xy = np.asarray(xy, dtype=float)
    assert xy.ndim == 2 and xy.shape[1] >= 2, "CSV pos deve avere almeno due colonne (x, y)"
    x = xy[:, 0]
    y = xy[:, 1]
    T = len(x)

    vx = np.zeros(T, dtype=float)
    vy = np.zeros(T, dtype=float)
    if T >= 2:
        vx[1:] = (x[1:] - x[:-1]) / dt
        vy[1:] = (y[1:] - y[:-1]) / dt

    ax = np.zeros(T, dtype=float)
    ay = np.zeros(T, dtype=float)
    if T >= 3:
        ax[2:] = (vx[2:] - vx[1:-1]) / dt
        ay[2:] = (vy[2:] - vy[1:-1]) / dt

    w_arr = np.full(T, w, dtype=float)
    h_arr = np.full(T, h, dtype=float)
    return np.stack([x, y, vx, vy, ax, ay, w_arr, h_arr], axis=1)

def load_tabular_csv(path):
    arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def detect_and_build(path, dt=1.0, x0=0.0, y0=0.0, w=1.0, h=1.0, as_pos=False):
    arr = load_tabular_csv(path)
    if arr.shape[1] >= 8:
        return arr[:, :8].astype(float)
    elif arr.shape[1] >= 2:
        xy = arr[:, :2].astype(float)
        if as_pos:
            return build_series_from_positions(xy, dt=dt, w=w, h=h)
        else:
            return build_series_from_displacements(xy, dt=dt, x0=x0, y0=y0, w=w, h=h)
    else:
        raise ValueError(f"{path}: formato non riconosciuto (servono >=2 o 8 colonne).")

# -------------------------
# Caricamento da cartelle
# -------------------------

def list_vehicle_ids(pred_dir: str, target_dir: str) -> List[int]:
    import fnmatch
    pred_ids = set()
    tgt_ids  = set()
    for fname in os.listdir(pred_dir):
        if fnmatch.fnmatch(fname.lower(), "pred_veicolo__*.csv"):
            try:
                vid = os.path.splitext(fname)[0].split("__")[-1]
                pred_ids.add(vid)
            except:  # noqa
                pass
    for fname in os.listdir(target_dir):
        if fnmatch.fnmatch(fname.lower(), "target_veicolo__*.csv"):
            try:
                vid = os.path.splitext(fname)[0].split("__")[-1]
                tgt_ids.add(vid)
            except:  # noqa
                pass
    return sorted(pred_ids & tgt_ids)

def load_series_from_dirs(pred_dir: str, target_dir: str, dt=1.0, x0=0.0, y0=0.0,
                          w=1.0, h=1.0, as_pos=False) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    ids = list_vehicle_ids(pred_dir, target_dir)
    print(ids)
    if not ids:
        raise FileNotFoundError("Nessuna coppia pred/target trovata nelle cartelle specificate.")

    pairs = []
    for vid in ids:
        pred_path = os.path.join(pred_dir,   f"pred_veicolo__{vid}.csv")
        tgt_path  = os.path.join(target_dir, f"target_veicolo__{vid}.csv")
        if not (os.path.exists(pred_path) and os.path.exists(tgt_path)):
            continue
        pr = detect_and_build(pred_path, dt=dt, x0=x0, y0=y0, w=w, h=h, as_pos=as_pos)
        gt = detect_and_build(tgt_path,  dt=dt, x0=x0, y0=y0, w=w, h=h, as_pos=as_pos)
        base = vid
        pairs.append((base, gt, pr))
    return pairs

# -------------------------
# Preparazione finestre
# -------------------------

def build_window_contexts(pairs: List[Tuple[str, np.ndarray, np.ndarray]],
                          scale=120, bg_color=(15, 18, 24)):
    """Crea origin, canvas e background per ogni veicolo."""
    contexts = []
    for base, gt, pr in pairs:
        series_dict = {f"{base}_GT": gt, f"{base}_Pred": pr}

        origin, (W, H), margin, scale_eff = make_canvas_params_multi(series_dict, scale=scale, margin_px=60)

        bg = np.full((H, W, 3), bg_color, dtype=np.uint8)
        step = max(1, int(1 * scale_eff))
        for x in range(margin, W - margin, step):
            cv2.line(bg, (x, margin), (x, H - margin), (40, 45, 55), 1)
        for y in range(margin, H - margin, step):
            cv2.line(bg, (margin, y), (W - margin, y), (40, 45, 55), 1)
        cv2.rectangle(bg, (margin, margin), (W - margin, H - margin), (70, 80, 95), 2)

        styles = {
            f"{base}_GT":   {"trail": (255, 255, 255), "box": (90, 200, 255),  "vec": (0, 255, 0),   "acc": (0, 140, 255)},
            f"{base}_Pred": {"trail": (120, 180, 255), "box": (140, 210, 255), "vec": (80, 200, 255),"acc": (50, 170, 255)},
        }

        maxT = max(len(gt), len(pr))

        contexts.append({
            "name": f"Veicolo {base}",
            "base": base,
            "series": series_dict,
            "origin": origin,
            "W": W, "H": H, "margin": margin, "scale": scale_eff,
            "bg": bg, "styles": styles, "maxT": maxT
        })
    return contexts

def tile_windows(contexts, cols_win=3, tile_gap=40):
    """Disposizione semplice in griglia sul desktop (best effort)."""
    if not contexts:
        return
    widths  = [c["W"] for c in contexts]
    heights = [c["H"] for c in contexts]
    med_w = int(np.median(widths))
    med_h = int(np.median(heights))
    dx = med_w + tile_gap
    dy = med_h + tile_gap
    for i, ctx in enumerate(contexts):
        x = (i % cols_win) * dx
        y = (i // cols_win) * dy
        try:
            cv2.moveWindow(ctx["name"], x, y)
        except:  # noqa
            pass

# -------------------------
# Visualizzazione: finestre multiple sincronizzate
# -------------------------

def visualize_multi_windows(pairs, scale=120, start_fps=30, cols_win=3, tile_gap=40):
    contexts = build_window_contexts(pairs, scale=scale)

    if not contexts:
        print("Nessun veicolo da visualizzare.")
        return

    # Crea le finestre
    for ctx in contexts:
        cv2.namedWindow(ctx["name"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctx["name"], ctx["W"], ctx["H"])

    tile_windows(contexts, cols_win=cols_win, tile_gap=tile_gap)

    t = 0
    maxT_global = max(ctx["maxT"] for ctx in contexts)
    playing = True
    speed = 1.0
    delay_ms = max(1, int(1000 / start_fps))

    show_trail, show_box, show_vectors = True, True, True

    while True:
        # Render di ogni finestra (sincronizzato sullo stesso t)
        for ctx in contexts:
            header = f"{ctx['name']}  frame {min(t+1, ctx['maxT'])}/{ctx['maxT']}"
            img = render_frame_multi(
                ctx["bg"], ctx["series"], t, ctx["origin"], ctx["margin"], ctx["scale"], ctx["styles"],
                show_trail=show_trail, show_box=show_box, show_vectors=show_vectors,
                show_legend=True, header_text=header
            )
            cv2.imshow(ctx["name"], img)

        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord(' '):        # Play/Pausa (tutte le finestre)
            playing = not playing
        elif key == ord('d'):      # Avanti di 1
            t = min(maxT_global - 1, t + 1)
        elif key == ord('a'):      # Indietro di 1
            t = max(0, t - 1)
        elif key in (ord('+'), ord('=')):
            speed = min(8.0, speed * 1.25)
        elif key == ord('-'):
            speed = max(0.125, speed / 1.25)
        elif key == ord('r'):
            t = 0
        elif key == ord('t'):
            show_trail = not show_trail
        elif key == ord('b'):
            show_box = not show_box
        elif key == ord('v'):
            show_vectors = not show_vectors
        elif key in (ord('q'), 27):  # q o ESC -> chiudi tutte
            break

        # Avanzamento automatico
        if playing:
            stepf = max(1, int(round(speed)))
            t = min(maxT_global - 1, t + stepf)
            if t >= maxT_global - 1:
                playing = False

        # Se l'utente chiude manualmente tutte le finestre, esci
        open_windows = sum(int(cv2.getWindowProperty(ctx["name"], cv2.WND_PROP_VISIBLE) > 0) for ctx in contexts)
        if open_windows == 0:
            break

    cv2.destroyAllWindows()

# -------------------------
# Demo / fallback singolo
# -------------------------

def _make_demo_pair(T=160, dt=0.05):
    base = np.array([0.5, 0.5,  2.0, 1.0,  0.1, 0.05,  1.2, 2.0], dtype=float)
    data_gt = np.zeros((T, 8), dtype=float)
    data_pr = np.zeros((T, 8), dtype=float)

    pos = base[:2].copy()
    vel = base[2:4].copy()
    acc0 = base[4:6].copy()
    w, h = base[6], base[7]

    pos_p = pos + np.array([0.02, -0.01])
    vel_p = vel * 0.98
    acc_p0 = acc0 * 1.02

    for t in range(T):
        acc = acc0 + 0.3 * np.array([np.sin(0.07*t), np.cos(0.05*t)])
        vel = vel + acc * dt
        pos = pos + vel * dt

        acc_p = acc_p0 + 0.3 * np.array([np.sin(0.07*t+0.12), np.cos(0.05*t-0.18)])
        vel_p = vel_p + acc_p * dt
        pos_p = pos_p + vel_p * dt

        data_gt[t] = [pos[0], pos[1], vel[0], vel[1], acc[0], acc[1], w, h]
        data_pr[t] = [pos_p[0], pos_p[1], vel_p[0], vel_p[1], acc_p[0], acc_p[1], w*0.98, h*1.02]

    return data_gt, data_pr

# -------------------------
# CLI
# -------------------------

def main():
    global INVERT_Y

    parser = argparse.ArgumentParser(description="Viewer GT/Pred multi-veicolo con finestre separate")
    # File singoli (per prova rapida)
    parser.add_argument("--gt", type=str, default=None, help="CSV Ground Truth (T,8) o (x,y)/(dx,dy)")
    parser.add_argument("--pred", type=str, default=None, help="CSV Predicted (T,8) o (x,y)/(dx,dy)")

    # Cartelle
    parser.add_argument("--pred-dir", type=str, default=None, help="Cartella con file pred_veicolo__NN.csv")
    parser.add_argument("--target-dir", type=str, default=None, help="Cartella con file target_veicolo__NN.csv")

    # Ricostruzione serie
    parser.add_argument("--dt", type=float, default=1.0, help="Intervallo tra campioni (s)")
    parser.add_argument("--x0", type=float, default=0.0, help="x iniziale (solo per dx,dy)")
    parser.add_argument("--y0", type=float, default=0.0, help="y iniziale (solo per dx,dy)")
    parser.add_argument("--w", type=float, default=1.0, help="larghezza box costante (per formati a 2 colonne)")
    parser.add_argument("--h", type=float, default=1.0, help="altezza box costante (per formati a 2 colonne)")
    parser.add_argument("--as-pos", action="store_true", help="Interpreta i file a 2 colonne come posizioni (x,y)")

    # Visual
    parser.add_argument("--scale", type=float, default=150, help="Pixel per unita' mondo")
    parser.add_argument("--fps", type=int, default=30, help="FPS playback")
    parser.add_argument("--invert-y", action="store_true", help="Forza INVERT_Y=True")
    parser.add_argument("--no-invert-y", action="store_true", help="Forza INVERT_Y=False")

    # Tiling finestre
    parser.add_argument("--cols-win", type=int, default=5, help="Colonne per disporre le finestre")
    parser.add_argument("--tile-gap", type=int, default=40, help="Spazio (px) tra finestre")

    args = parser.parse_args()

    if args.invert_y:
        INVERT_Y = True
    if args.no_invert_y:
        INVERT_Y = False

    # --- Modalità cartelle: una finestra per veicolo ---
    if args.pred_dir and args.target_dir:
        pairs = load_series_from_dirs(args.pred_dir, args.target_dir,
                                      dt=args.dt, x0=args.x0, y0=args.y0,
                                      w=args.w, h=args.h, as_pos=args.as_pos)
        visualize_multi_windows(pairs, scale=args.scale, start_fps=args.fps,
                                cols_win=args.cols_win, tile_gap=args.tile_gap)
        return

    # --- File singoli (apre una sola finestra, utile per test) ---
    if args.gt and args.pred:
        gt = detect_and_build(args.gt,  dt=args.dt, x0=args.x0, y0=args.y0, w=args.w, h=args.h, as_pos=args.as_pos)
        pr = detect_and_build(args.pred, dt=args.dt, x0=args.x0, y0=args.y0, w=args.w, h=args.h, as_pos=args.as_pos)
        pairs = [("test", gt, pr)]
        visualize_multi_windows(pairs, scale=args.scale, start_fps=args.fps,
                                cols_win=1, tile_gap=args.tile_gap)
        return

    # --- Demo fallback ---
    gt, pr = _make_demo_pair()
    pairs = [("demo", gt, pr)]
    visualize_multi_windows(pairs, scale=args.scale, start_fps=args.fps,
                            cols_win=1, tile_gap=args.tile_gap)

if __name__ == "__main__":
    main()

"""
Uso tipico (una finestra per veicolo):
  python viewer.py --pred-dir Predictions --target-dir Targets
"""
