from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from collections import Counter

app = Flask(__name__)

OPTION_COLS_MAP = {
    "Kuesioner FACT-L": 5,
    "Kuesioner WHODAS 2.0": 5,
    "Kuesioner MSPSS": 7,
    "Kuesioner ESAS-r": 11,
    "Kuesioner HADS-D": 4,   # sesuaikan jika instrumen Anda beda
}

# ==== PARAMETER TUNING CEPAT ====
Y_TOL = 28                 # toleransi grouping baris (px)
MARGIN_RATIO = 0.08        # margin crop ROI (lebih kecil supaya centang tidak kepotong)
LINES_DILATE_ITER = 1      # jangan terlalu besar (kalau 2 sering “makan” centang)

ABS_THR = 1.6              # ambang minimal persen tinta untuk dianggap berisi
DIFF_THR = 0.6             # selisih minimal best vs second (persen tinta)
# ===============================

def resize_if_large(img, max_w=1600):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def cluster_rows(cells, y_tol=Y_TOL):
    if not cells:
        return []

    cells = sorted(cells, key=lambda b: b[1])
    rows = []

    for (x, y, w, h) in cells:
        cy = y + h / 2.0
        placed = False
        for r in rows:
            if abs(cy - r["cy"]) < y_tol:
                r["cells"].append((x, y, w, h))
                r["cy"] = (r["cy"] * (len(r["cells"]) - 1) + cy) / len(r["cells"])
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "cells": [(x, y, w, h)]})

    rows = sorted(rows, key=lambda r: r["cy"])
    for r in rows:
        r["cells"] = sorted(r["cells"], key=lambda b: b[0])

    return [r["cells"] for r in rows]

def mode_total_cols(rows, min_cols):
    counts = [len(r) for r in rows if len(r) >= min_cols]
    if not counts:
        return None
    c = Counter(counts)
    return c.most_common(1)[0][0]

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def compute_cell_percent(binary_img, box, margin_ratio=MARGIN_RATIO):
    x, y, w, h = box
    mw = int(w * margin_ratio)
    mh = int(h * margin_ratio)

    x1, y1 = x + mw, y + mh
    x2, y2 = x + w - mw, y + h - mh

    roi = binary_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    tinta = cv2.countNonZero(roi)  # putih = tinta (karena INV)
    luas = roi.shape[0] * roi.shape[1]
    return (tinta / luas) * 100 if luas > 0 else 0.0

def pick_checkbox_like_cells(cells):
    """
    Ambil sel yang mirip kotak pilihan:
    - aspect ratio mendekati 1
    - ukuran tidak ekstrem
    """
    if not cells:
        return []

    cand = []
    for (x, y, w, h) in cells:
        r = w / float(h + 1e-6)
        area = w * h
        # filter dasar: harus “hampir kotak” dan area masuk akal
        if 0.65 <= r <= 1.55 and 300 <= area <= 120000:
            cand.append((x, y, w, h))

    if len(cand) < 10:
        return cand  # fallback

    ws = np.array([b[2] for b in cand], dtype=np.float32)
    hs = np.array([b[3] for b in cand], dtype=np.float32)
    med_w = float(np.median(ws))
    med_h = float(np.median(hs))

    filtered = []
    for (x, y, w, h) in cand:
        if (0.55 * med_w <= w <= 1.8 * med_w) and (0.55 * med_h <= h <= 1.8 * med_h):
            filtered.append((x, y, w, h))

    return filtered if len(filtered) >= 10 else cand

@app.route("/scan", methods=["POST"])
def scan():
    if "file" not in request.files:
        return jsonify({"status": "error", "pesan": "Tidak ada file (key harus 'file')."}), 400

    file = request.files["file"]
    jenis_kuesioner = request.form.get("jenis_kuesioner", "").strip()
    option_cols = OPTION_COLS_MAP.get(jenis_kuesioner, 5)

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "pesan": "Gambar tidak valid / gagal dibaca."}), 400

        img = resize_if_large(img, 1600)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = clahe_gray(gray)  # bantu untuk foto gelap/bayangan

        # Threshold untuk ekstraksi struktur + tinta
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )

        # Ekstrak garis tabel
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
        lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
        table_structure = cv2.add(lines_h, lines_v)

        # Hapus garis dari threshold agar centang lebih “bersih”
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_mask = cv2.dilate(table_structure, k, iterations=LINES_DILATE_ITER)
        thresh_clean = cv2.bitwise_and(thresh, cv2.bitwise_not(lines_mask))

        # Temukan kontur sel
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 20 < w < 900 and 20 < h < 500:
                cells.append((x, y, w, h))

        if not cells:
            return jsonify({"status": "error", "pesan": "Tidak ada sel terdeteksi. Pastikan garis tabel terlihat jelas."}), 422

        # Fokus ke sel kotak pilihan (square-like)
        checkbox_cells = pick_checkbox_like_cells(cells)
        if len(checkbox_cells) < option_cols * 3:
            # fallback ke semua sel kalau terlalu sedikit
            checkbox_cells = cells

        rows = cluster_rows(checkbox_cells, y_tol=Y_TOL)
        rows = [r for r in rows if len(r) >= option_cols]

        if not rows:
            return jsonify({"status": "error", "pesan": "Baris tabel tidak terbentuk. Foto mungkin terlalu miring/terpotong."}), 422

        # mode jumlah kolom per baris (untuk buang noise)
        total_cols_mode = mode_total_cols(rows, min_cols=option_cols)
        if total_cols_mode is None:
            total_cols_mode = option_cols

        jawaban_per_soal = []
        debug_rows = []
        total_baris_diproses = 0
        soal_nomor = 0

        for r in rows:
            # filter noise
            if abs(len(r) - total_cols_mode) > 3 and len(r) > total_cols_mode:
                continue

            # Ambil opsi: biasanya kotak pilihan berada di bagian kanan,
            # tapi karena kita sudah ambil "checkbox-like", ini relatif aman.
            r_sorted = sorted(r, key=lambda b: b[0])
            opsi_cells = r_sorted[-option_cols:]  # ambil paling kanan

            if len(opsi_cells) != option_cols:
                continue

            total_baris_diproses += 1

            pers = [compute_cell_percent(thresh_clean, box, MARGIN_RATIO) for box in opsi_cells]
            best_idx = int(np.argmax(pers))
            pers_sorted = sorted(pers, reverse=True)
            best = pers_sorted[0]
            second = pers_sorted[1] if len(pers_sorted) > 1 else 0.0

            # Kriteria berisi (lebih longgar dibanding sebelumnya)
            # + dynamic check: best juga harus lebih tinggi dari rata-rata baris secara jelas
            mean = float(np.mean(pers))
            std = float(np.std(pers))
            dyn_thr = mean + 2.0 * std  # adaptif per baris

            is_marked = (best > max(ABS_THR, dyn_thr)) and ((best - second) > DIFF_THR)

            debug_rows.append({
                "persentase_opsi": [round(x, 2) for x in pers],
                "mean": round(mean, 2),
                "std": round(std, 2),
                "dyn_thr": round(dyn_thr, 2),
                "best_idx": best_idx + 1,
                "best": round(best, 2),
                "second": round(second, 2),
                "is_marked": bool(is_marked),
            })

            if is_marked:
                soal_nomor += 1
                jawaban_per_soal.append({
                    "soal_nomor": soal_nomor,
                    "jawaban": f"Opsi {best_idx + 1}",
                    "persentase_pilihan": round(best, 2),
                    "persentase_opsi": [round(x, 2) for x in pers],
                })

        return jsonify({
            "status": "sukses",
            "jenis_kuesioner": jenis_kuesioner,
            "option_cols": option_cols,
            "total_sel_terdeteksi": len(cells),
            "total_checkbox_like": len(checkbox_cells),
            "total_baris_soal_diproses": total_baris_diproses,
            "total_jawaban_terdeteksi": len(jawaban_per_soal),
            "jawaban_per_soal": jawaban_per_soal,
            "debug_rows": debug_rows
        })

    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
