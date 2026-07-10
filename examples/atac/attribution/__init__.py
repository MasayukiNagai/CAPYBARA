"""Tier-B interpretation for the CAPY ATAC benchmark.

Attribution / contribution scoring + TF-MoDISco motif discovery, mirroring
ChromBPNet's Tier B so the only changed variable is the model. The contribution
scorers here write ChromBPNet's exact ``.h5`` schema (``raw``/``shap``/
``projected_shap``, each ``(N, 4, L)``), so the container's ``modisco``
(``modisco motifs`` / ``modisco report``) runs on the output unchanged.

Two engines, **user-selectable at launch** (``--method``):
  * ``gradientshap`` (``captum``) — **default for both models**. Robust to the
    accessibility net's ``hybrid_attention`` bottleneck, and preferred on the bias
    net too because DeepLIFT shows high convergence deltas on CAPY's pooling U-Net.
  * ``deeplift`` (``tangermeme.deep_lift_shap``) — closest to ChromBPNet's DeepSHAP;
    opt-in via ``--method deeplift`` for the bias model (its local ``residual_conv``
    bottleneck is DeepLIFT-clean) and for comparison. Not recommended for the
    attention-based nobias model.
The profile/count scalar targets and the ``.h5`` schema are identical across both
engines, so downstream ``modisco`` is unchanged; only score generation differs.
"""
