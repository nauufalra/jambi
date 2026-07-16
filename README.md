# Project JAMBI 🛡️
**Judicial Automated Monitoring for Burglary Intrusion**

Sistem keamanan mandiri berbasis webcam laptop. Saat mendeteksi **manusia**, sistem otomatis membunyikan alarm keras dan merekam video sebagai barang bukti. Dirancang efisien untuk dijalankan semalaman (8+ jam) tanpa membuat laptop kepanasan.

---

## 1. Panduan Instalasi

### a. Prasyarat
- Python 3.9 – 3.12 (kamu memakai 3.12 ✔)
- Webcam bawaan laptop yang berfungsi

### b. (Disarankan) Buat virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### c. Install library yang dibutuhkan
```powershell
pip install ultralytics opencv-python pygame
```
> Catatan:
> - `ultralytics` sudah otomatis membawa `torch` (versi CPU) dan YOLOv8.
> - Model `yolov8n.pt` (~6 MB) akan **terunduh otomatis** saat pertama kali dijalankan — pastikan ada internet di run pertama saja.

### d. (Opsional tapi disarankan) Siapkan file alarm
Letakkan sebuah file `alarm.mp3` di folder yang sama dengan `dorm_guard.py`.
Jika file tidak ada, sistem tetap jalan memakai **beep darurat** dari terminal.

> 💡 Pilih MP3 sirene/alarm yang keras & berdurasi beberapa detik (akan diputar berulang otomatis).

---

## 2. Cara Menjalankan

```powershell
python dorm_guard.py
```

- Jendela pratinjau akan muncul dengan status **AMAN** (hijau) / **BAHAYA** (merah).
- Tekan **`q`** di jendela, atau **`Ctrl+C`** di terminal, untuk berhenti dengan aman.
- Rekaman bukti tersimpan di folder **`recordings/`** dengan nama `JAMBI_intrusion_<tanggal>_<jam>.avi`.
- Log aktivitas tercatat di **`jambi_activity.log`**.

### Konfigurasi cepat (edit class `Config` di dalam `dorm_guard.py`)
| Parameter | Default | Fungsi |
|---|---|---|
| `CAMERA_INDEX` | `0` | Ganti bila webcam bawaan bukan index 0 |
| `CONFIDENCE_THRESHOLD` | `0.42` | Turunkan (→0.40) bila kamar sangat gelap; naikkan bila banyak alarm palsu |
| `COOLDOWN_SECONDS` | `5.0` | Berapa detik "sepi" sebelum alarm & rekaman berhenti |
| `TARGET_FPS` | `10` | Batas FPS pemrosesan (jangan dinaikkan agar laptop tetap adem) |
| `SHOW_PREVIEW` | `True` | Set `False` untuk mode headless (lebih hemat resource) |

---

## 3. Tips Operasional (Agar Laptop Tidak "Sleep" Semalaman)

Ini krusial — kalau laptop masuk mode Sleep, pengawasan mati total.

### A. Cegah Sleep lewat Settings (paling mudah)
`Settings → System → Power & battery → Screen and sleep`
- **When plugged in, put my device to sleep after → Never**
- **When plugged in, turn my screen off after → Never** (atau biarkan layar mati, itu tidak menghentikan skrip)

### B. Cegah Sleep sementara lewat PowerShell (praktis, satu perintah)
Jalankan **sebelum** memulai skrip, biarkan jendela ini terbuka semalaman:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
> `0` = Never. Untuk mengembalikan besok pagi, ganti `0` dengan mis. `30` (menit).

### C. Poin penting lain
- 🔌 **Colokkan charger.** Jangan andalkan baterai untuk 8 jam + AI + webcam.
- 🌡️ **Beri ventilasi.** Letakkan laptop di permukaan keras (bukan kasur/bantal) agar panas terbuang. Pulse throttling ~10 FPS sudah menekan suhu, tapi aliran udara tetap penting.
- 🔊 **Cek volume sistem** dalam kondisi maksimal, dan uji alarm sekali sebelum tidur (lambaikan tangan ke kamera).
- 💡 **Jangan 100% gelap.** Webcam laptop tanpa night-vision butuh sedikit cahaya (lampu tidur redup / cahaya dari jendela) agar siluet manusia terdeteksi.
- 🖥️ Layar boleh mati — yang penting **jangan Sleep/Hibernate**. Skrip tetap memproses walau layar padam.

---

## 4. Cara Kerja Singkat (Arsitektur)

```
[Webcam] → [Frame Ingestion CV2] → [Pulse Throttle ~10 FPS]
                                          │
                                          ▼
[Simpan .avi] ← [Video Writer] ← [YOLOv8 Nano (Class 0: person)]
                                          │
                                          ▼
[Alarm Audio Non-blocking] ←──── [Pygame Mixer Engine]
```

- **Pulse Throttling:** membatasi pemrosesan ke ~10 FPS → hemat CPU 60%+ → cegah *thermal throttling*.
- **Non-blocking Audio:** alarm diputar via `pygame.mixer` di thread terpisah → webcam tidak nge-freeze.
- **Event-driven Recording:** rekaman hanya jalan saat ada manusia → hemat storage.
- **Cooldown:** alarm & rekaman berhenti otomatis setelah area sepi `COOLDOWN_SECONDS`.
- **Fallback:** bila `alarm.mp3` hilang/rusak → sistem beralih ke beep darurat + log jelas.

---

## 5. Troubleshooting

| Gejala | Solusi |
|---|---|
| `Webcam gagal dibuka` | Tutup Zoom/Teams/aplikasi lain yang memakai kamera; cek izin kamera Windows |
| Alarm tak bersuara, hanya beep | `alarm.mp3` tidak ada/rusak — taruh file MP3 yang valid |
| Banyak alarm palsu | Naikkan `CONFIDENCE_THRESHOLD` ke `0.45`, tambah sedikit cahaya |
| Manusia tak terdeteksi di gelap | Turunkan ke `0.40` + tambah lampu tidur redup |
| Laptop panas | Pastikan `TARGET_FPS = 10`, beri ventilasi, jangan taruh di kasur |
```
