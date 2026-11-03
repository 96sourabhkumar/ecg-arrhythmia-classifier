"""
config.py — Project constants for ECG arrhythmia classification on FPGA.

WRITE ONCE. NEVER EDIT.

Every record list and class mapping in this file comes from a published
protocol. If you change them, your results stop being comparable with the
literature and your inter-patient claim becomes unverifiable.

Sources:
  DS1/DS2 split ....... de Chazal, P., O'Dwyer, M. & Reilly, R. B.,
                        "Automatic Classification of Heartbeats Using ECG
                        Morphology and Heartbeat Interval Features",
                        IEEE Trans. Biomedical Engineering, 2004.
                        Restated in arXiv:2503.07276 (2025).
  AAMI mapping ........ ANSI/AAMI EC57, as applied in the above.
  Dataset spec ........ https://physionet.org/content/mitdb/1.0.0/
  Pan-Tompkins ........ Pan, J. & Tompkins, W. J., IEEE TBME BME-32(3):230-236,
                        1985. DOI 10.1109/TBME.1985.325532
"""

# ---------------------------------------------------------------------------
# DATASET  [PROVEN - PhysioNet MIT-BIH database description]
# ---------------------------------------------------------------------------

FS_ORIGINAL = 360          # Hz, MIT-BIH native sampling rate
ADC_BITS    = 11           # 11-bit resolution over a 10 mV range
ADC_RANGE_MV = 10.0
LEAD_INDEX  = 0            # channel 0 = MLII in most records

# [CALCULATED] 10 mV / 2^11 = 4.88 uV per ADC step
LSB_MV = ADC_RANGE_MV / (2 ** ADC_BITS)


# ---------------------------------------------------------------------------
# INTER-PATIENT SPLIT  [PROVEN - de Chazal et al. 2004]
#
# No patient appears in both lists. This is what makes the evaluation honest.
# Randomly shuffling individual beats instead causes PATIENT LEAKAGE and
# inflates accuracy by ~5-10 points. See Complete Guide Part 11.
# ---------------------------------------------------------------------------

# Records containing paced beats, excluded to comply with ANSI/AAMI
EXCLUDED_RECORDS = [102, 104, 107, 217]

# Training set
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

# Test set - DO NOT TOUCH UNTIL FINAL EVALUATION
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# [CALCULATED] 22 + 22 + 4 = 48, matching the database size.
assert len(DS1) == 22, "DS1 must contain 22 records"
assert len(DS2) == 22, "DS2 must contain 22 records"
assert len(EXCLUDED_RECORDS) == 4
assert not set(DS1) & set(DS2), "PATIENT LEAKAGE: DS1 and DS2 overlap"
assert len(DS1) + len(DS2) + len(EXCLUDED_RECORDS) == 48


# ---------------------------------------------------------------------------
# AAMI 5-CLASS MAPPING  [PROVEN - ANSI/AAMI EC57]
# ---------------------------------------------------------------------------

AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
CLASS_INDEX  = {c: i for i, c in enumerate(AAMI_CLASSES)}

SYMBOL_TO_AAMI = {
    # N - Normal and bundle branch block beats
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S - Supraventricular ectopic  (hardest class; timing matters more than shape)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V - Ventricular ectopic  (clinically the most dangerous)
    'V': 'V', 'E': 'V',
    # F - Fusion
    'F': 'F',
    # Q - Unknown / paced
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

CLASS_DESCRIPTION = {
    'N': 'Normal / bundle branch block',
    'S': 'Supraventricular ectopic',
    'V': 'Ventricular ectopic',
    'F': 'Fusion',
    'Q': 'Unknown / paced',
}


# ---------------------------------------------------------------------------
# SIGNAL PROCESSING  [PROPOSED - see Complete Guide Part 10.13]
#
# We decimate 360 -> 180 Hz by exactly 2 (halfband FIR, the cheapest
# resampler that exists). Pan-Tompkins was designed for 200 Hz; at 180 Hz
# the cutoffs shift from 11/5 Hz to 9.9/4.5 Hz, still inside the QRS band.
#
# [MUST MEASURE] whether this costs detection sensitivity. Test 360 vs 180
# against the annotations and report both - it turns a design choice into
# an experimental result.
# ---------------------------------------------------------------------------

DECIMATION  = 2
FS          = FS_ORIGINAL // DECIMATION        # 180 Hz

# Pan-Tompkins timing constants  [PROVEN - Pan & Tompkins 1985]
MWI_WINDOW_MS      = 150    # moving-window integration
REFRACTORY_MS      = 200    # physiological lockout after a beat
TWAVE_WINDOW_MS    = 360    # T-wave discrimination window
SEARCHBACK_PERCENT = 166    # of average RR before search-back triggers

# [CALCULATED] convert to samples at FS = 180 Hz
MWI_WINDOW_SAMPLES   = int(MWI_WINDOW_MS   * FS / 1000)   # 27
REFRACTORY_SAMPLES   = int(REFRACTORY_MS   * FS / 1000)   # 36
TWAVE_WINDOW_SAMPLES = int(TWAVE_WINDOW_MS * FS / 1000)   # 64

# [PROPOSED] Use 32 not 27 for the MWI window so the divide becomes >> 5.
# Costs a 178 ms window instead of 150 ms.
# [MUST MEASURE] the sensitivity impact - this is a real experiment.
MWI_WINDOW_HW = 32
MWI_SHIFT     = 5


# ---------------------------------------------------------------------------
# BEAT WINDOW  [PROPOSED - see Complete Guide Part 11.1]
# 128 is chosen because it is 2^7: a 7-bit address counter wraps for free.
# ---------------------------------------------------------------------------

BEAT_WINDOW    = 128
BEAT_PRE       = 45     # samples before the R peak
BEAT_POST      = 83     # samples after  (45 + 83 = 128)

# [CALCULATED] 128/180 = 711 ms total; 250 ms before, 456 ms after the peak
assert BEAT_PRE + BEAT_POST == BEAT_WINDOW


# ---------------------------------------------------------------------------
# CNN  [PROPOSED - see Complete Guide Part 13.8 for the full derivation]
# ---------------------------------------------------------------------------

CNN_LAYERS = [
    # (in_ch, out_ch, kernel, out_length)
    (1,   8, 5, 128),
    (8,  16, 5,  64),
    (16, 32, 3,  32),
    (32, 32, 3,  16),
]
NUM_RR_FEATURES = 4
NUM_CLASSES     = 5

def _derive():
    """Recompute the headline numbers so they can never silently drift."""
    params = macs = 0
    for (ic, oc, k, L) in CNN_LAYERS:
        params += ic * oc * k + oc
        macs   += oc * L * k * ic
    gap_out = CNN_LAYERS[-1][1]                       # 32
    fc_in   = gap_out + NUM_RR_FEATURES               # 36
    params += fc_in * NUM_CLASSES + NUM_CLASSES       # 185
    macs   += fc_in * NUM_CLASSES                     # 180
    return params, macs

TOTAL_PARAMS, TOTAL_MACS = _derive()

# These are the numbers quoted throughout the project documents.
assert TOTAL_PARAMS == 5561, f"expected 5561 params, got {TOTAL_PARAMS}"
assert TOTAL_MACS   == 144564, f"expected 144564 MACs, got {TOTAL_MACS}"


# ---------------------------------------------------------------------------
# QUANTIZATION  [PROPOSED - see Complete Guide Part 15]
# ---------------------------------------------------------------------------

WEIGHT_BITS = 8
ACT_BITS    = 8
ACC_BITS    = 32
# Restrict to -127..+127 so the range is symmetric; avoids a class of bugs.
INT8_MIN, INT8_MAX = -127, 127
SATURATE = True     # ALWAYS saturate, NEVER wrap. Non-negotiable.


# ---------------------------------------------------------------------------
# HARDWARE  [PROVEN - AMD/Xilinx DS190, DS180, UG473]
# ---------------------------------------------------------------------------

CLOCK_HZ = 100_000_000
MAC_UNITS = 16          # P; sweep 1/4/8/16/32 in Experiment E12

DEVICES = {
    'xc7z020': {'lut': 53_200, 'ff': 106_400, 'dsp': 220, 'bram36': 140,
                'board': 'PYNQ-Z2 / Zybo Z7-20', 'has_arm': True},
    'xc7a35t': {'lut': 20_800, 'ff':  41_600, 'dsp':  90, 'bram36':  50,
                'board': 'Basys3', 'has_arm': False},
}
BRAM36_BYTES = 4608     # 36 kbit


def report():
    """Print the headline numbers. Run this to sanity-check the design."""
    w = TOTAL_PARAMS - sum(oc for (_, oc, _, _) in CNN_LAYERS) - NUM_CLASSES
    b = TOTAL_PARAMS - w
    model_bytes = w * 1 + b * 4
    cycles = TOTAL_MACS / MAC_UNITS
    infer_s = cycles / CLOCK_HZ
    beat_s = 60 / 75

    print(f"Sampling         : {FS_ORIGINAL} Hz -> {FS} Hz (decimate {DECIMATION})")
    print(f"Beat window      : {BEAT_WINDOW} samples = {1000*BEAT_WINDOW/FS:.0f} ms")
    print(f"Parameters       : {TOTAL_PARAMS:,}")
    print(f"MACs per beat    : {TOTAL_MACS:,}")
    print(f"Model size INT8  : {model_bytes:,} bytes")
    print(f"BRAM36 for model : {-(-model_bytes // BRAM36_BYTES)}")
    print(f"Inference @ P={MAC_UNITS}  : {infer_s*1e6:.1f} us ({cycles:,.0f} cycles)")
    print(f"Beat interval    : {beat_s*1000:.0f} ms")
    print(f"Duty cycle       : {100*infer_s/beat_s:.4f} %")
    print(f"Idle             : {100*(1-infer_s/beat_s):.2f} %")
    print(f"Margin at P=1    : {beat_s / (TOTAL_MACS/CLOCK_HZ):.0f}x real time")


if __name__ == '__main__':
    report()
