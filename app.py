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
        # 1. BACA GAMBAR
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 2. UBAH KE HITAM PUTIH (THRESHOLDING)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gunakan thresholding yang lebih adaptif agar kebal terhadap bayangan HP
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. CARI SEMUA BENTUK (KONTUR) DI KERTAS
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        kotak_ditemukan = []
        
        # 4. SARING HANYA YANG BENTUKNYA KOTAK
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rasio = w / float(h)
            
            # ATURAN KOTAK: 
            # Lebar & Tinggi antara 20 - 100 piksel (abaikan titik debu atau garis panjang)
            # Rasio antara 0.8 sampai 1.2 (bentuknya mendekati persegi/bujur sangkar)
                if 20 < w < 1000 and 20 < h < 1000 and 0.8 <= rasio <= 1.2:
                kotak_ditemukan.append((x, y, w, h))
                
        # 5. URUTKAN KOTAK DARI ATAS KE BAWAH (Berdasarkan kordinat Y)
        # Ini agar kotak Soal No 1 dibaca lebih dulu dari Soal No 2
        kotak_ditemukan = sorted(kotak_ditemukan, key=lambda b: b[1])
        
        # 6. PERIKSA ISI MASING-MASING KOTAK
        hasil_scan = []
        for urutan, (x, y, w, h) in enumerate(kotak_ditemukan):
            # Potong (crop) hanya area kotak tersebut
            area_kotak = thresh[y:y+h, x:x+w]
            
            # Hitung tintanya
            jumlah_piksel_tinta = cv2.countNonZero(area_kotak)
            persentase = (jumlah_piksel_tinta / (w * h)) * 100
            
            status_kotak = "terisi" if persentase > 15 else "kosong" # Batas toleransi dinaikkan sedikit untuk full page
            
            # Simpan hasil kotak ini ke daftar
            hasil_scan.append({
                "soal_nomor": urutan + 1,
                "status": status_kotak,
                "persentase": round(persentase, 2)
            })
            
        # 7. KEMBALIKAN SEMUA DATA KE PHP
        return jsonify({
            "status": "sukses",
            "total_kotak_terdeteksi": len(kotak_ditemukan),
            "data_jawaban": hasil_scan
        })
        
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run()
