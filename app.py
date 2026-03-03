from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from collections import Counter

app = Flask(__name__)

# Mapping jumlah opsi jawaban (kolom pilihan) per instrumen
OPTION_COLS_MAP = {
    "Kuesioner FACT-L": 5,
    "Kuesioner WHODAS 2.0": 5,
    "Kuesioner MSPSS": 7,
    "Kuesioner ESAS-r": 11,
    "Kuesioner HADS-D": 4,   # sesuaikan jika instrumen Anda beda
}

def resize_if_large(img, max_w=1600):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def cluster_rows(cells, y_tol=25):
    """
    Kelompokkan bounding box sel menjadi baris berdasarkan center-y.
    cells: list of (x,y,w,h)
    return: list of rows, each row: list of (x,y,w,h) sorted left->right
    """
    if not cells:
        return []

    cells_sorted = sorted(cells, key=lambda b: b[1])
    rows = []

    for (x, y, w, h) in cells_sorted:
        cy = y + h / 2.0
        placed = False

        for r in rows:
            if abs(cy - r["cy"]) < y_tol:
                r["cells"].append((x, y, w, h))
                # update cy incremental
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
    """
    Ambil jumlah kolom total yang paling sering muncul (mode) untuk baris yang masuk akal.
    Ini membantu mendeteksi apakah ada kolom 'No/Item' di kiri.
    """
    counts = [len(r) for r in rows if len(r) >= min_cols]
    if not counts:
        return None
    c = Counter(counts)
    return c.most_common(1)[0][0]

def compute_cell_percent(thresh_clean, box, margin_ratio=0.10):
    x, y, w, h = box
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)

    x1 = x + margin_w
    y1 = y + margin_h
    x2 = x + w - margin_w
    y2 = y + h - margin_h

    roi = thresh_clean[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    tinta = cv2.countNonZero(roi)  # putih = tinta (karena THRESH_BINARY_INV)
    luas = roi.shape[0] * roi.shape[1]
    return (tinta / luas) * 100 if luas > 0 else 0.0

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

        img = resize_if_large(img, max_w=1600)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold agar tahan bayangan
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Ekstraksi garis tabel
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
        lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
        table_structure = cv2.add(lines_h, lines_v)

        # Hilangkan garis dari thresh supaya perhitungan tinta fokus ke centang
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_mask = cv2.dilate(table_structure, k, iterations=2)
        thresh_clean = cv2.bitwise_and(thresh, cv2.bitwise_not(lines_mask))

        # Cari kontur sel
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # filter ukuran (sesuaikan jika form Anda sangat kecil/besar)
            if 30 < w < 800 and 20 < h < 400:
                cells.append((x, y, w, h))

        if not cells:
            return jsonify({
                "status": "error",
                "pesan": "Tidak ada sel terdeteksi. Coba foto lebih lurus/lebih dekat, atau pastikan garis tabel terlihat."
            }), 422

        # Filter tambahan untuk buang kotak raksasa (border luar)
        ws = np.array([b[2] for b in cells], dtype=np.float32)
        hs = np.array([b[3] for b in cells], dtype=np.float32)
        med_w = float(np.median(ws))
        med_h = float(np.median(hs))

        filtered = []
        for (x, y, w, h) in cells:
            if (0.55 * med_w <= w <= 1.8 * med_w) and (0.55 * med_h <= h <= 1.8 * med_h):
                filtered.append((x, y, w, h))

        cells = filtered if len(filtered) >= option_cols * 3 else cells  # fallback kalau terlalu sedikit

        # Kelompokkan menjadi baris
        rows = cluster_rows(cells, y_tol=25)
        rows = [r for r in rows if len(r) >= option_cols]  # baris yang masuk akal

        if not rows:
            return jsonify({
                "status": "error",
                "pesan": "Baris tabel tidak terbentuk. Pastikan foto menampilkan tabel secara utuh dan tidak terlalu miring."
            }), 422

        # Deteksi total kolom per baris (mode) lalu ambil opsi di sisi kanan (biasanya kolom jawaban)
        total_cols_mode = mode_total_cols(rows, min_cols=option_cols)
        if total_cols_mode is None:
            total_cols_mode = option_cols

        # Jawaban per soal
        jawaban_per_soal = []
        debug_rows = []

        soal_nomor = 0
        total_baris_diproses = 0

        for r in rows:
            # Abaikan baris yang jauh berbeda dari mode (noise)
            if abs(len(r) - total_cols_mode) > 2 and len(r) > total_cols_mode:
                continue

            # Ambil kolom paling kanan sesuai jumlah opsi
            r_sorted = sorted(r, key=lambda b: b[0])
            opsi_cells = r_sorted[-option_cols:]

            if len(opsi_cells) != option_cols:
                continue

            total_baris_diproses += 1

            pers = [compute_cell_percent(thresh_clean, box, margin_ratio=0.10) for box in opsi_cells]
            median = float(np.median(pers))
            adj = [p - median for p in pers]

            best_idx = int(np.argmax(adj))
            adj_sorted = sorted(adj, reverse=True)
            best = adj_sorted[0]
            second = adj_sorted[1] if len(adj_sorted) > 1 else -999.0

            # Kriteria "terisi" (tuning aman untuk menghindari header/teks tercetak):
            # 1) harus lebih gelap dari median baris
            # 2) harus beda jelas dibanding opsi kedua
            # 3) raw persentase juga harus cukup
            raw_best = pers[best_idx]
            is_marked = (best > 3.0) and ((best - second) > 1.5) and (raw_best > 6.0)

            debug_rows.append({
                "persentase_opsi": [round(x, 2) for x in pers],
                "median_row": round(median, 2),
                "adj": [round(x, 2) for x in adj],
                "best_idx": best_idx + 1,
                "raw_best": round(raw_best, 2),
                "is_marked": bool(is_marked),
            })

            if is_marked:
                soal_nomor += 1
                jawaban_per_soal.append({
                    "soal_nomor": soal_nomor,
                    "jawaban": f"Opsi {best_idx + 1}",
                    "persentase_pilihan": round(raw_best, 2),
                    "persentase_opsi": [round(x, 2) for x in pers],
                    "skor_rel": round(best, 2),
                })

        return jsonify({
            "status": "sukses",
            "jenis_kuesioner": jenis_kuesioner,
            "option_cols": option_cols,
            "total_sel_terdeteksi": len(cells),
            "total_baris_soal_diproses": total_baris_diproses,
            "total_jawaban_terdeteksi": len(jawaban_per_soal),
            "jawaban_per_soal": jawaban_per_soal,
            "debug_rows": debug_rows  # boleh dipakai untuk tuning; kalau tidak perlu, bisa dihapus
        })

    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
