# results/models/inter_patient_int8/

INT8 export of the inter-patient model (`../ecgnet_inter_patient.pt`), same file formats as
`hardware/weights/` (see that README). It gives 94.60 % on DS2 with `software/evaluate.py`.
To run this model on the board, copy these files to `hardware/weights/` and rebuild the bitstream
(the weights are stored in on-chip memory). That has not been done; the board was run with the
patient-adaptive weights only.
