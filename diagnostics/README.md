# FCN depth diagnostics (clean paper nets)

Diagnostics of how the LI=CA gate shapes the representation *through depth*, on the clean-repo
paper FCN (`fcn_mnist_tanh_ns_ki_d10`, and `_d100` where noted). Run with
`PYTHONPATH=. python diagnostics/<script>.py`.

- **fcn_depth_diagnostics.py** — combined figure: (1) LOGIT-LENS (trained classifier applied to each
  intermediate layer → accuracy vs depth) and (2) SCALE SEPARATION (across-layer correlation time τ
  per spatial-frequency band; SSI = τ_coarse/τ_fine). FCN-10 MNIST, LI=CA vs none. → `fcn_depth_diagnostics.png`
- **fcn_scale_separation.py** — full scale-separation diagnostic, d10 (CA vs none) + d100 (CA, long
  depth axis): C_b(k) correlation curves, τ-vs-frequency, per-scale velocity. → `scratchpad/fcn_scale_separation.png`

Key result (MNIST): LI=CA keeps the class decodable in the classifier frame from layer 1 (62%→96%)
while LI=none is near-chance until the last layer; and LI=CA has a slow coarse carrier (τ≈9 layers at
d10, ≈30 at d100) with fast fine churn (τ≈1), SSI 18–29, vs none SSI≈1.25. The two coincide: the slow
coarse scale IS the class-decodable carrier the logit-lens reads through depth.
