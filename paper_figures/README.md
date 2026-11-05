# Paper figures

One script per figure of the manuscript (and the activation video). Every script writes its figure to `out/` under the
file name the manuscript uses (`\includegraphics{figures/<name>}`), so `out/` can be copied into `manuscript/figures/`.
Run from the repository root: `PYTHONPATH=. python paper_figures/<script>.py`.

Most scripts read the trained networks in `paper_runs/` directly. Some read a cache written by an analysis script in
`paper_code/`; run that first if the cache is missing (column "needs").

| Manuscript figure (label) | Output (`out/`) | Script | Needs |
|---|---|---|---|
| Overview of the mechanism (`fig:block_diagram`) | `FIG1.pdf` | hand-drawn; copy in `static/` | – |
| MNIST rule sweep (`fig:mnist_rule_sweep_metrics`) | `rule_sweep_300ep.png` | `rule_sweep_300ep_figure.py` | `diagnostics/fcn_coffee_rules.json` (`paper_code/fcn_coffee_rules.py`) |
| Activation maps and classifier weights (`fig:act_maps`) | `activation_classifier-fcn.png` | `fcn_depth_activation_classifier.py` | – |
| Depth diagnostics (`fig:depth_diag`) | `fcn_depth_diagnostics.png` | `fcn_depth_diagnostics.py` | – |
| Seed agreement by scale (`fig:seed_agreement`) | `fcn_seed_agreement.png` | `fcn_seed_agreement_paper_figure.py` | outputs of `paper_code/fcn_seed_agreement_scales.py` (r0, r1, r2, fine) |
| Coherence for ten rules (`fig:rule_coherence`) | `fcn_rule_coherence.png` | `fcn_rule_coherence.py` (also writes `diagnostics/fcn_rule_coherence.json`, the numbers of the text) | – |
| Saliency on and off the object (`fig:saliency`) | `saliency_onoff_separation.png` | `saliency_onoff_separation.py` (annotates the signed log-ratio) | – |
| Saliency per image, saliency appendix (`fig:saliency_dist`) | `saliency_signed_distributions.pdf` | `saliency_signed_distributions.py` | `diagnostics/saliency_signed_perimage.npz` (`paper_code/saliency_signed_metrics.py`) |
| Alternatives to a "1" (`fig:perdigit`) | `perdigit_probability.png` | `one_vs_digit_panels_absolute.py` | `diagnostics/one_vs_digit_leaderboard.json` (`paper_code/one_vs_digit_leaderboard.py`; plots the temperature-calibrated gaps `gap_cal_*`) |
| Noise resilience (`fig:noise_resilience`) | `Noise_ressiliance.png` | `paper_noise_resilience.py` | `paper_runs/_noise_per_type.json` (`paper_code/noise_per_type.py`), `paper_runs/_fcn_matched_noise.json` (`paper_code/fcn_matched_noise.py`) |
| Decoy patch (`fig:decoy`) | `decoy_patch_5seed.png` | `decoy_patch_figure.py` | output of `paper_code/decoy_patch_check.py` |
| Gates and blend, formulation appendix (`fig:gate_blend`) | `fig_gates_B5S3_3panel.png` | `gates_figure.py` (model-free) | – |
| Architectures appendix | `architecture_diagrams.pdf` | `architecture_diagrams.py` (model-free) | – |
| Noise breakdown by type, noise appendix (`fig:nb_fcn10`, `fig:nb_fcn2`, `fig:nb_cnngray`, `fig:nb_cnncolour`) | `noise_breakdown_{fcn10,fcn2,cnn_gray,cnn_colour}.png` | `noise_breakdown_figures.py` | `paper_runs/_noise_per_type.json`, `paper_runs/_noise_uniform_blend.json` (`paper_code/noise_per_type.py --uniform`) |
| Noise examples, noise appendix | `noise_examples.png` | `noise_examples_figure.py` | – |
| Glider gun, sanity-check appendix (`fig:glider_gun_continuous`) | `FIG6.png` | made with the messy repo (`viz/make_fig5.py`) at an earlier version of its gate; copy in `static/` | – |
| Input blur, controls appendix (`fig:blur_maps`) | `fcn_blur_activation_classifier.png` | `fcn_blur_activation_classifier.py` | the control networks (`paper_code/fcn_controls.py`) |
| Saliency of the blur controls (`fig:blursal`) | `blur_saliency_cnn.png` | `blur_saliency.py` | the control networks (`paper_code/cnn_controls.py`) |
| Where lateral inhibition operates (`fig:occupancy`) | `occupancy_std.png` | `mechinterp_occ_std.py` | – |
| Noise resilience along training (snapshots appendix) | `fcn2_snapshot_drop.png` | `fcn2_snapshot_drop.py` | `paper_runs/_fcn_matched_noise.json`, the snapshots (`scripts/run_snapshots.py`) |
| Co-activation graph, Section 4.2 (`fig:netsci`) | `netsci_results.png` | `netsci_results.py` | `diagnostics/percolation_epochs.json`, `percolation_blur.json` (`paper_code/percolation_epochs.py`, `percolation_blur.py`); layouts via `paper_code/percolation_geometry.py` |
| Percolation over training, percolation appendix (`fig:percolation`) | `percolation_epochs.png` | `percolation_epochs_pair.py` | `diagnostics/percolation_epochs.json` (`paper_code/percolation_epochs.py`, on the snapshots of `configs/percol_epochs/`) |
| Activation video (doi.org/10.6084/m9.figshare.34750569) | `fcn10_mnist_cube_digits_0to9.mp4` | `activation_video.py` (runs `activation_cube.py` per digit, joins with ffmpeg) | – |

## Numbers of the review revision (2026-11)

Tables and numbers added in answer to the review come from these analysis scripts in `paper_code/` (each writes a
JSON in `diagnostics/`):

| Manuscript | Script |
|---|---|
| Signed saliency metrics, Section 4.3 and `tab:salchecks` | `saliency_signed_metrics.py` |
| Signed saliency of the blur and gate-route controls (`tab:blursal`, gate-route table) | `saliency_signed_controls.py` |
| Surrogate input gradient of the image normalisation (`app:perchgrad`, `tab:perchgrad`) | `perch_stat_gradient.py` |
| Ranking of the alternatives (`app:ranking`, `tab:ranking`) | `alternative_ranking.py` |
| Scale-separation index, every value in the paper (`app:ssi`, `tab:ssi`) | `ssi_all.py`, on the one definition in `scale_separation_index.py` |
| Uncertainty of the noise gains (`app:uncertainty`, `tab:uncertainty`) | `noise_uncertainty.py` |
| Percolation readouts, `tab:percolation` (k_c, giant at 8, largest jump, edge and floor overlap) | `percolation_epochs.py`, `percolation_blur.py`, `percolation_checks.py`, `percolation_epochs_summary.py` |
| Image-plane geometry, random-geometric-graph reference (`app:percolation`) | `percolation_geometry.py` |
| r_c, strong-pair locality, site percolation, fixed-survivor test (`app:percolation`) | `percolation_operating_point.py` |
| Scale-free and small-world tests (`app:percolation`, Section 4.2) | `percolation_smallworld.py` |
| Isolated nodes on active units, grid-rule and collapse appendices | `isolates_dead_units.py` (paper FCNs, 4 datasets), `isolates_dead_units_controls.py` (MNIST controls, 3 seeds) |
| Gate-route share of the gradient (`app:steering`) | `gate_steering_measure.py` |
| Networks retrained with every gate route cut (`tab:steering`) | `gate_detached_train.py` (MLP and CNN: on the paper configs; WRN: `configs/gate_detached/wrn_cifar10_100ep_ca.json`; modes `full` and `detached`), then `gate_detached_eval.py` |
| CNN seed agreement and weight averaging of the CNN and WRN (Discussion) | `cnn_seed_agreement.py`, `weight_average_cnn_wrn.py` |

The symmetric score max(on/off, off/on) of `saliency_multiseed.py`, `saliency_causal_checks.py`,
`cnn_controls_5seed.py` and `deep_vs_shallow_blur.py` is reported in the paper only as "region separability".
