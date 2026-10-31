# software/qrs/

Golden models of the QRS hardware. Both are pure integer NumPy code and are bit-exact with the RTL.

| File | Hardware it specifies | Main functions |
|---|---|---|
| `pan_tompkins.py` | `hardware/rtl/qrs/{lowpass,highpass,derivative,moving_window_integrator,peak_detector}.v` | `decimate_fixed` (360→180 Hz, 11-tap halfband), `lowpass_fixed`, `highpass_fixed`, `derivative_fixed`, `square_fixed`, `mwi_fixed`, `detect_peaks_fixed`, `pan_tompkins_fixed` |
| `rr_features.py` | `hardware/rtl/qrs/rr_feature_engine.v` | `constants(s_rr)`, `rr_int8(peaks, i, K, clip)`, `rr_int8_stream` |
