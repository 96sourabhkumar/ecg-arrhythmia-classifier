"""
gen_filter_vectors.py — dump golden-model stage I/O as Verilog $readmemh vectors.

    python gen_filter_vectors.py

For every front-end stage, writes the exact input stream and the exact expected
output stream. The Verilog testbenches read these and compare sample by sample.
A mismatch on ANY sample is a failure — "close" is not acceptable.

Values are written as 32-bit two's-complement hex, 8 digits, one per line.

Outputs to hardware/sim/vectors/:
    filter_input.mem        decimated 180 Hz samples (input to lowpass)
    filter_lowpass_expected.mem     expected lowpass output
    filter_highpass_expected.mem    expected highpass output
    filter_derivative_expected.mem       expected derivative output
    filter_square_expected.mem      expected squarer output
    filter_mwi_expected.mem         expected moving-window-integrator output
    filter_meta.txt    sample count and source record
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qrs.pan_tompkins as pt
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'mitdb')
VEC  = os.path.join(HERE, '..', 'hardware', 'sim', 'vectors')
os.makedirs(VEC, exist_ok=True)

N_SAMPLES = 20000        # ~111 s at 180 Hz. Long enough to exercise everything.
RECORD    = 100


def w32(path, arr):
    """Write int array as 32-bit two's-complement hex, one per line."""
    with open(path, 'w') as f:
        for v in arr:
            f.write(format(int(v) & 0xFFFFFFFF, '08x') + '\n')


def w64(path, arr):
    with open(path, 'w') as f:
        for v in arr:
            f.write(format(int(v) & 0xFFFFFFFFFFFFFFFF, '016x') + '\n')


def main():
    import wfdb
    p = os.path.join(RAW, str(RECORD))
    if not os.path.exists(p + '.dat'):
        print(f"FAIL: {p}.dat missing — run mitbih.py --download")
        sys.exit(1)

    rec = wfdb.rdrecord(p)
    ch  = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
    adc = np.round(rec.p_signal[:, ch] * rec.adc_gain[ch]).astype(np.int64)
    print(f"record {RECORD}, channel {ch} ({rec.sig_name[ch]}), "
          f"{len(adc)} samples @ 360 Hz")

    # ---- run the golden model, capturing every stage --------------------
    dec = pt.decimate_fixed(adc)
    lp  = pt.lowpass_fixed(dec)
    hp  = pt.highpass_fixed(lp)
    dv  = pt.derivative_fixed(hp)
    sq  = pt.square_fixed(dv)
    mw  = pt.mwi_fixed(sq)

    n = min(N_SAMPLES, len(dec))
    print(f"emitting {n} samples per vector file\n")

    # ---- range check: make sure our chosen Verilog widths are adequate ---
    def bits_needed(a):
        m = int(max(abs(int(a.min())), abs(int(a.max()))))
        return m.bit_length() + 1          # +1 for sign

    stages = [('decimated', dec[:n], 32), ('lowpass', lp[:n], 32),
              ('highpass', hp[:n], 32), ('derivative', dv[:n], 32),
              ('squarer', sq[:n], 64), ('mwi', mw[:n], 64)]
    print(f"  {'stage':<12}{'min':>14}{'max':>14}{'bits':>7}{'width':>7}  fits")
    ok = True
    for name, arr, width in stages:
        b = bits_needed(arr)
        fits = b <= width
        ok &= fits
        print(f"  {name:<12}{int(arr.min()):>14}{int(arr.max()):>14}"
              f"{b:>7}{width:>7}  {'yes' if fits else 'NO — OVERFLOW'}")
    if not ok:
        print("\nFAIL: a stage needs more bits than the Verilog provides")
        sys.exit(1)

    # ---- write vectors ---------------------------------------------------
    w32(os.path.join(VEC, 'filter_input.mem'),     dec[:n])
    w32(os.path.join(VEC, 'filter_lowpass_expected.mem'),  lp[:n])
    w32(os.path.join(VEC, 'filter_highpass_expected.mem'), hp[:n])
    w32(os.path.join(VEC, 'filter_derivative_expected.mem'),    dv[:n])
    w64(os.path.join(VEC, 'filter_square_expected.mem'),   sq[:n])
    w64(os.path.join(VEC, 'filter_mwi_expected.mem'),      mw[:n])

    with open(os.path.join(VEC, 'filter_meta.txt'), 'w') as f:
        f.write(f"record {RECORD}\nsamples {n}\nfs {pt.FS}\n")

    print(f"\nwrote 6 vector files to hardware/sim/vectors/")
    print(f"PASS — now run:  cd ..\\simulation  then  .\\run_tests.ps1")


if __name__ == '__main__':
    main()
