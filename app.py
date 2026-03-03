from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan_kuesioner():
    if 'file' not in request.files:
        return jsonify({"status": "error", "pesan": "Tidak ada file"}), 400
    
    file = request.files['file']
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Thresholding balik (Kertas hitam, garis putih)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Ekstrak Garis Horizontal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # 3. Ekstrak Garis Vertikal
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # 4. Gabungkan untuk membuat Mask Tabel
        table_mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)
        table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # 5. Cari Kontur Sel dalam Tabel
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Saring ukuran sel (sesuaikan dengan ukuran kolom skala likert Anda)
            if 30 < w < 200 and 20 < h < 100:
                cells.append((x, y, w, h))
        
        # 6. Urutkan Sel: Atas ke Bawah, lalu Kiri ke Kanan
        # Kita beri toleransi y/10 agar sel dalam satu baris dianggap memiliki y yang sama
        cells = sorted(cells, key=lambda b: (b[1] // 10, b[0]))
        
        hasil_scan = []
        for i, (x, y, w, h) in enumerate(cells):
            # Crop bagian dalam sel (margin 5px agar tidak kena garis tabel)
            cell_roi = thresh[y+5:y+h-5, x+5:x+w-5]
            
            tinta = cv2.countNonZero(cell_roi)
            persentase = (tinta / (cell_roi.shape[0] * cell_roi.shape[1])) * 100 if cell_roi.size > 0 else 0
            
            hasil_scan.append({
                "cell_index": i + 1,
                "status": "terisi" if persentase > 8 else "kosong",
                "persentase": round(persentase, 2)
            })
            
        return jsonify({
            "status": "sukses",
            "total_sel_terdeteksi": len(cells),
            "data_jawaban": hasil_scan
        })
        
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run()
