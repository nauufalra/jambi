#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 Project JAMBI - Judicial Automated Monitoring for Burglary Intrusion
=====================================================================

Sistem keamanan mandiri berbasis webcam laptop untuk pengawasan kamar
asrama semalaman. Ketika sistem mendeteksi kehadiran MANUSIA (person),
sistem akan:

    1. Membunyikan alarm suara (.mp3) berulang, volume maksimal.
    2. Merekam video kejadian secara otomatis (barang bukti / judicial footprint).
    3. Berhenti otomatis (cooldown) saat manusia tidak lagi terdeteksi.

Arsitektur:
    [Webcam] -> [Frame Ingestion CV2] -> [Pulse Throttle ~10 FPS]
                                              |
                                              v
    [Simpan .avi] <- [Video Writer] <- [YOLOv8 Nano (Class 0: person)]
                                              |
                                              v
    [Alarm Audio Non-blocking] <----- [Pygame Mixer Engine]

Author : Senior Software Engineer (dibuat untuk Rendi)
Tanggal: 2026-07-16
=====================================================================
"""

import os
import sys
import time
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# IMPORT DEPENDENSI EKSTERNAL
# Dibungkus try/except agar pesan error-nya ramah bila library belum terpasang.
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    sys.exit("[FATAL] OpenCV belum terpasang. Jalankan: pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[FATAL] Ultralytics belum terpasang. Jalankan: pip install ultralytics")

# pygame bersifat opsional: bila gagal, sistem tetap jalan dengan fallback beep.
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ===========================================================================
# KONFIGURASI GLOBAL
# Semua parameter yang mungkin ingin kamu ubah dikumpulkan di sini agar rapi.
# ===========================================================================
class Config:
    # --- Sumber Video ---
    CAMERA_INDEX = 0                 # 0 = webcam bawaan laptop. Ganti bila pakai kamera lain.

    # --- Model AI ---
    MODEL_PATH = "yolov8n.pt"        # YOLOv8 Nano: ringan, cukup akurat di CPU.
    PERSON_CLASS_ID = 0              # Di dataset COCO, class 0 = "person".

    # Confidence threshold untuk kondisi remang/gelap (tanpa night vision).
    # 0.40-0.45 -> sensitif terhadap siluet manusia, tapi menahan false positive
    # akibat noise pixel di area gelap.
    CONFIDENCE_THRESHOLD = 0.42

    # --- Pulse Throttling (Efisiensi Energi) ---
    # Batasi pemrosesan AI ke ~10 FPS untuk mencegah thermal throttling
    # saat berjalan 8+ jam. Menghemat CPU 60%+ dibanding 30 FPS konstan.
    TARGET_FPS = 10
    FRAME_INTERVAL = 1.0 / TARGET_FPS   # detik per frame (~0.1 s)

    # --- Cooldown ---
    # Berapa detik manusia harus HILANG dari pandangan sebelum alarm & rekaman
    # dimatikan. Mencegah alarm mati-nyala saat orang bergerak sebentar keluar frame.
    COOLDOWN_SECONDS = 5.0

    # --- Audio Alarm ---
    ALARM_FILE = "alarm.mp3"         # File alarm eksternal. Bila hilang -> fallback beep.
    ALARM_VOLUME = 1.0               # 0.0 - 1.0 (1.0 = maksimal).

    # --- Perekaman Video ---
    RECORDINGS_DIR = "recordings"    # Folder output rekaman.
    VIDEO_CODEC = "XVID"             # XVID + .avi: hemat CPU untuk kompresi real-time.
    VIDEO_EXTENSION = ".avi"
    RECORDING_FPS = 10.0             # FPS metadata untuk file video (samakan dgn TARGET_FPS).

    # --- Tampilan ---
    SHOW_PREVIEW = True              # True = tampilkan jendela pratinjau (tekan 'q' untuk keluar).
                                     # Set False bila ingin benar-benar headless/hemat resource.


# ===========================================================================
# SETUP LOGGING
# Log ke konsol DAN ke file agar ada jejak kejadian sepanjang malam.
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("jambi_activity.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("JAMBI")


# ===========================================================================
# KOMPONEN 1: MESIN ALARM (Non-blocking Audio)
# ===========================================================================
class AlarmEngine:
    """
    Mengelola pemutaran audio alarm secara NON-BLOCKING menggunakan pygame.mixer.

    Non-blocking artinya: proses pemutaran audio berjalan di thread internal
    milik pygame, sehingga loop pengambilan frame webcam TIDAK ikut membeku
    (freeze/stutter) saat alarm berbunyi.

    Bila file alarm.mp3 tidak ada / rusak / pygame tidak tersedia, sistem
    otomatis memakai FALLBACK berupa system beep ('\\a') + log peringatan.
    """

    def __init__(self, config: Config):
        self.config = config
        self.is_playing = False          # Status apakah alarm sedang aktif.
        self.use_fallback = False        # True bila terpaksa pakai beep fallback.
        self._mixer_ready = False
        self._last_fallback_beep = 0.0   # Timestamp beep fallback terakhir (throttle).

        self._init_mixer()

    def _init_mixer(self):
        """Inisialisasi pygame mixer + validasi file audio."""
        if not _PYGAME_AVAILABLE:
            logger.warning("pygame tidak tersedia -> memakai fallback beep sistem.")
            self.use_fallback = True
            return

        try:
            # Inisialisasi khusus mixer saja (tidak perlu seluruh subsystem pygame).
            pygame.mixer.init()
            self._mixer_ready = True
        except Exception as exc:
            logger.error("Gagal inisialisasi pygame mixer (%s) -> fallback beep.", exc)
            self.use_fallback = True
            return

        # Validasi keberadaan & integritas file alarm.
        if not os.path.isfile(self.config.ALARM_FILE):
            logger.warning(
                "File alarm '%s' TIDAK ditemukan -> fallback beep sistem.",
                self.config.ALARM_FILE,
            )
            self.use_fallback = True
            return

        try:
            pygame.mixer.music.load(self.config.ALARM_FILE)
            pygame.mixer.music.set_volume(self.config.ALARM_VOLUME)
            logger.info("Alarm audio '%s' siap digunakan.", self.config.ALARM_FILE)
        except Exception as exc:
            # File ada tapi rusak/format tidak didukung.
            logger.error("File alarm rusak/tidak didukung (%s) -> fallback beep.", exc)
            self.use_fallback = True

    def start(self):
        """Mulai membunyikan alarm (berulang tanpa henti). Idempotent."""
        if self.is_playing:
            return  # Sudah berbunyi, jangan tumpuk.

        self.is_playing = True
        logger.critical(">>> ALARM AKTIF: kehadiran manusia terdeteksi! <<<")

        if self.use_fallback:
            self._fallback_beep(force=True)
            return

        try:
            # loops=-1 -> putar berulang tanpa batas sampai di-stop.
            pygame.mixer.music.play(loops=-1)
        except Exception as exc:
            logger.error("Gagal memutar alarm (%s) -> beralih ke fallback.", exc)
            self.use_fallback = True
            self._fallback_beep(force=True)

    def stop(self):
        """Hentikan alarm dengan aman. Idempotent."""
        if not self.is_playing:
            return

        self.is_playing = False
        logger.info("Alarm dimatikan (cooldown / area kembali aman).")

        if self.use_fallback:
            return

        try:
            pygame.mixer.music.stop()
        except Exception as exc:
            logger.error("Gagal menghentikan alarm dengan bersih (%s).", exc)

    def tick(self):
        """
        Dipanggil tiap iterasi loop utama.
        Untuk mode fallback (tanpa mp3), kita perlu memicu beep secara berkala
        karena '\\a' hanya berbunyi sekali per pemanggilan.
        """
        if self.is_playing and self.use_fallback:
            self._fallback_beep()

    def _fallback_beep(self, force: bool = False):
        """Beep darurat via terminal bell, di-throttle agar tidak spam ekstrem."""
        now = time.time()
        # Beep tiap ~1 detik saja supaya konsol tidak dibanjiri.
        if force or (now - self._last_fallback_beep) >= 1.0:
            self._last_fallback_beep = now
            # '\a' = ASCII Bell -> memicu beep sistem pada sebagian besar terminal.
            sys.stdout.write("\a")
            sys.stdout.flush()
            logger.warning("[FALLBACK ALARM] BEEP! Manusia terdeteksi - bangun & periksa!")

    def cleanup(self):
        """Bersihkan resource mixer saat program berakhir."""
        self.stop()
        if self._mixer_ready:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


# ===========================================================================
# KOMPONEN 2: PEREKAM VIDEO (Event-driven Recording)
# ===========================================================================
class VideoRecorder:
    """
    Mengelola siklus perekaman video berbasis kejadian (event-driven).

    Rekaman HANYA aktif saat manusia terdeteksi -> hemat storage, tidak
    merekam sepanjang malam. File ditutup dengan aman (release) saat cooldown
    agar tidak korup.
    """

    def __init__(self, config: Config, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.writer = None
        self.current_file = None
        self.is_recording = False

        # Pastikan folder output tersedia.
        os.makedirs(self.config.RECORDINGS_DIR, exist_ok=True)

    def start(self):
        """Buat file rekaman baru dengan timestamp. Idempotent."""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"JAMBI_intrusion_{timestamp}{self.config.VIDEO_EXTENSION}"
        filepath = os.path.join(self.config.RECORDINGS_DIR, filename)

        try:
            fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)
            self.writer = cv2.VideoWriter(
                filepath,
                fourcc,
                self.config.RECORDING_FPS,
                (self.frame_width, self.frame_height),
            )

            if not self.writer.isOpened():
                logger.error("VideoWriter gagal dibuka untuk '%s'. Rekaman dibatalkan.", filepath)
                self.writer = None
                return

            self.current_file = filepath
            self.is_recording = True
            logger.info("Mulai merekam bukti -> %s", filepath)
        except Exception as exc:
            logger.error("Gagal memulai perekaman (%s).", exc)
            self.writer = None

    def write(self, frame):
        """Tulis satu frame ke file (hanya bila sedang merekam)."""
        if self.is_recording and self.writer is not None:
            try:
                self.writer.write(frame)
            except Exception as exc:
                logger.error("Gagal menulis frame video (%s).", exc)

    def stop(self):
        """Tutup file rekaman dengan aman. Idempotent."""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.writer is not None:
            try:
                self.writer.release()   # Penting: release agar file .avi tidak korup.
                logger.info("Rekaman disimpan & ditutup dengan aman -> %s", self.current_file)
            except Exception as exc:
                logger.error("Gagal menutup file rekaman dengan bersih (%s).", exc)
            finally:
                self.writer = None
                self.current_file = None

    def cleanup(self):
        """Pastikan writer dilepas saat program berakhir."""
        self.stop()


# ===========================================================================
# KOMPONEN 3: SISTEM UTAMA JAMBI
# ===========================================================================
class JambiGuardian:
    """Orkestrator utama: menyatukan kamera, AI, alarm, dan perekam."""

    def __init__(self, config: Config):
        self.config = config
        self.cap = None
        self.model = None
        self.alarm = None
        self.recorder = None

        # State machine deteksi.
        self.human_present = False           # Apakah manusia sedang terdeteksi?
        self.last_detection_time = 0.0       # Timestamp deteksi manusia terakhir.

    # ----------------------------------------------------------------------
    # INISIALISASI
    # ----------------------------------------------------------------------
    def _init_camera(self):
        """Buka webcam dengan penanganan error."""
        logger.info("Membuka webcam (index %d)...", self.config.CAMERA_INDEX)

        # CAP_DSHOW mempercepat pembukaan kamera di Windows & mengurangi warning.
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Webcam (index {self.config.CAMERA_INDEX}) gagal dibuka. "
                "Pastikan kamera tidak dipakai aplikasi lain & izin kamera aktif."
            )

        # Uji baca satu frame untuk memastikan kamera benar-benar mengalirkan data.
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam terbuka tapi gagal membaca frame. Cek driver kamera.")

        logger.info("Webcam berhasil dibuka.")

    def _init_model(self):
        """Muat model YOLOv8 Nano. Otomatis diunduh bila belum ada."""
        logger.info("Memuat model YOLOv8 Nano ('%s')...", self.config.MODEL_PATH)
        try:
            self.model = YOLO(self.config.MODEL_PATH)
            logger.info("Model AI siap.")
        except Exception as exc:
            raise RuntimeError(
                f"Gagal memuat model '{self.config.MODEL_PATH}' ({exc}). "
                "Pastikan koneksi internet aktif saat pertama kali (untuk auto-download)."
            )

    def setup(self):
        """Inisialisasi seluruh komponen sebelum loop utama berjalan."""
        self._init_camera()
        self._init_model()

        # Ambil dimensi frame aktual dari kamera untuk konfigurasi VideoWriter.
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.alarm = AlarmEngine(self.config)
        self.recorder = VideoRecorder(self.config, width, height)

        logger.info("=== Project JAMBI SIAP MENGAWASI (%dx%d) ===", width, height)

    # ----------------------------------------------------------------------
    # DETEKSI
    # ----------------------------------------------------------------------
    def _detect_person(self, frame):
        """
        Jalankan YOLO pada satu frame.
        Return: (terdeteksi_manusia: bool, frame_beranotasi)
        """
        # Hanya proses class 0 (person) untuk kecepatan; verbose=False agar konsol bersih.
        results = self.model(
            frame,
            classes=[self.config.PERSON_CLASS_ID],
            conf=self.config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        person_found = False
        annotated = frame

        if results:
            result = results[0]
            # Bila ada minimal satu bounding box, berarti manusia terdeteksi.
            if result.boxes is not None and len(result.boxes) > 0:
                person_found = True
            # plot() menggambar bounding box untuk pratinjau & rekaman bukti.
            if self.config.SHOW_PREVIEW or self.recorder.is_recording:
                annotated = result.plot()

        return person_found, annotated

    def _handle_state(self, person_found: bool, frame):
        """
        State machine: kelola transisi AMAN <-> BAHAYA berdasarkan deteksi
        + logika cooldown.
        """
        now = time.time()

        if person_found:
            self.last_detection_time = now
            if not self.human_present:
                # --- TRANSISI: AMAN -> BAHAYA ---
                self.human_present = True
                self.alarm.start()
                self.recorder.start()
        else:
            if self.human_present:
                # Manusia sempat terdeteksi tapi kini hilang -> cek durasi cooldown.
                elapsed = now - self.last_detection_time
                if elapsed >= self.config.COOLDOWN_SECONDS:
                    # --- TRANSISI: BAHAYA -> AMAN ---
                    self.human_present = False
                    self.alarm.stop()
                    self.recorder.stop()

        # Selama berstatus bahaya, terus tulis frame ke rekaman & jaga alarm hidup.
        if self.human_present:
            self.recorder.write(frame)
            self.alarm.tick()   # Diperlukan untuk mode fallback beep berkala.

    # ----------------------------------------------------------------------
    # LOOP UTAMA
    # ----------------------------------------------------------------------
    def run(self):
        """Loop pengawasan utama dengan pulse throttling ~10 FPS."""
        logger.info("Mulai pengawasan. Tekan Ctrl+C (atau 'q' di jendela) untuk berhenti.")
        consecutive_read_failures = 0

        try:
            while True:
                loop_start = time.time()

                # 1) Ambil frame dari webcam.
                ok, frame = self.cap.read()
                if not ok:
                    # Toleransi gangguan sesaat; berhenti bila gagal terus-menerus.
                    consecutive_read_failures += 1
                    logger.warning("Gagal membaca frame (%d kali berturut).", consecutive_read_failures)
                    if consecutive_read_failures >= 30:
                        logger.error("Webcam tidak responsif. Menghentikan sistem.")
                        break
                    time.sleep(0.1)
                    continue
                consecutive_read_failures = 0

                # 2) Deteksi manusia via YOLO.
                person_found, annotated = self._detect_person(frame)

                # 3) Kelola alarm & rekaman berdasarkan hasil deteksi.
                #    Simpan frame beranotasi (ada bounding box) sebagai bukti.
                self._handle_state(person_found, annotated)

                # 4) Pratinjau opsional.
                if self.config.SHOW_PREVIEW:
                    status = "BAHAYA - MANUSIA!" if self.human_present else "AMAN"
                    color = (0, 0, 255) if self.human_present else (0, 255, 0)
                    cv2.putText(
                        annotated, f"JAMBI: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                    )
                    cv2.imshow("Project JAMBI - Dorm Guard", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Tombol 'q' ditekan. Menghentikan sistem.")
                        break

                # 5) PULSE THROTTLING: tidur secukupnya agar total ~10 FPS.
                #    Ini kunci efisiensi energi & pencegah thermal throttling.
                elapsed = time.time() - loop_start
                sleep_time = self.config.FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interupsi keyboard (Ctrl+C) diterima. Mematikan sistem...")
        finally:
            self.cleanup()

    # ----------------------------------------------------------------------
    # PEMBERSIHAN
    # ----------------------------------------------------------------------
    def cleanup(self):
        """Lepas semua resource dengan aman agar tidak ada file korup / kamera terkunci."""
        logger.info("Membersihkan resource...")
        if self.alarm:
            self.alarm.cleanup()
        if self.recorder:
            self.recorder.cleanup()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("=== Project JAMBI dimatikan. Selamat beristirahat. ===")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main():
    config = Config()
    guardian = JambiGuardian(config)
    try:
        guardian.setup()
        guardian.run()
    except RuntimeError as exc:
        # Error inisialisasi kritis (kamera/model) -> laporkan jelas lalu keluar.
        logger.error("GAGAL START: %s", exc)
        guardian.cleanup()
        sys.exit(1)
    except Exception as exc:
        logger.exception("Error tak terduga: %s", exc)
        guardian.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
