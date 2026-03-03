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
        
        # Thresholding adaptif untuk menangani bayangan pada kertas fisik
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Ekstraksi struktur tabel (Morfologi)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
        
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
        
        table_structure = cv2.add(lines_h, lines_v)
        
        # Temukan kontur sel
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter ukuran sel agar tidak mengambil teks di luar tabel
            if 30 < w < 500 and 20 < h < 200:
                cells.append((x, y, w, h))
        
        # FIX: Pengurutan dengan toleransi 50 pixel agar satu baris tidak pecah
        cells = sorted(cells, key=lambda b: (int(b[1] / 60), b[0]))
        
        hasil_scan = []
        for i, (x, y, w, h) in enumerate(cells):
            # Crop area tengah sel (Margin 15% untuk menghindari garis hitam tabel)
            margin_w = int(w * 0.15)
            margin_h = int(h * 0.15)
            cell_roi = thresh[y+margin_h : y+h-margin_h, x+margin_w : x+w-margin_w]
            
            tinta = cv2.countNonZero(cell_roi)
            luas = cell_roi.shape[0] * cell_roi.shape[1]
            persentase = (tinta / luas) * 100 if luas > 0 else 0
            
            # Sensitivitas ditingkatkan menjadi 10% agar lebih akurat
            status = "terisi" if persentase > 10 else "kosong"
            
            hasil_scan.append({
                "cell_index": i + 1,
                "status": status,
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
