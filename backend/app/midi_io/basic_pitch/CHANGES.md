# Changes to the Basic Pitch model

`nmp.onnx` is the unmodified model file from basic-pitch 0.4.0 (Copyright 2022 Spotify AB, Apache License 2.0, see `LICENSE` and `NOTICE`).

`nmp_cowrite.onnx` is a modified version made by the Cowrite project (2026-11-20):

- Same graph, inputs and outputs as `nmp.onnx`. Only the weights and biases of the last six convolution layers (16,700 parameters) differ. The CQT front end is unchanged.
- Those layers were fine-tuned for 800 steps on synthetic sample-pack style loops: randomly generated chords, arpeggios, melodies and bass lines, rendered with simple synthesizers and FluidR3 GM samples (Frank Wen, CC BY 3.0), with reverb, delay, chorus and saturation applied.
- Training code: `backend/train/transcriber/` in this repository.
