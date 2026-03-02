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
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        jumlah_piksel_tinta = cv2.countNonZero(thresh)
        tinggi, lebar = thresh.shape
        persentase = (jumlah_piksel_tinta / (tinggi * lebar)) * 100
        
        status_kotak = "terisi" if persentase > 5 else "kosong"
        
        return jsonify({"status": "sukses", "hasil": status_kotak, "persentase": round(persentase, 2)})
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run()
