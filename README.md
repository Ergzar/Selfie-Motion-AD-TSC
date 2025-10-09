# Selfie Motion Anomaly Detection & Time Series Classification

This repo contains code and data for detecting anomalies and classifying individual users via short windows of smartphone motion during selfie capture in the Candour ID application.

## What’s here
- **Benchmarks**
  - `AD/` — anomaly detection scripts (oneclass_learning_user.py & user-vs-spoof.py)
  - `TSC/` — time series classification script (user_classification.py)
- **Dataset (`CanSelfie`)**
  - Two multivariate UCR-style `.ts` files:
    - `CanSelfie_START200.ts` — **4 s** (200 samples @ 50 Hz) starting **at** `SELFIE_START` (camera opened)
    - `CanSelfie_CAPTURE200.ts` — **1 s before + 3 s after** `SELFIE_CAPTURE`
  - Each sample has **12 channels**:
    - accelerometer `acc_` (x,y,z)  
    - linear acceleration `lin_acc_` (x,y,z)  
    - gyroscope `gyro_` (x,y,z)  
    - magnetometer `magnet_` (x,y,z)
  - Raw data (`dataset_raw/`)
    - Full-length sequences plus event timestamps in `touch/touch.json`.
    - Additional sensors such as Android `rotationVector` and `gravity` (note varying sampling rates).

