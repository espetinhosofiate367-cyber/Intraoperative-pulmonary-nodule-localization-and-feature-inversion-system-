# Size-Controlled UMAP Summary

Global UMAP was intentionally demoted because the full latent space mixes strong size and weaker depth effects.
The figure below controls the size condition and visualizes depth structure within selected size bands.

- Size `0.25 cm`: `n=73`, out-of-fold probe accuracy `0.9041`
- Size `1.00 cm`: `n=84`, out-of-fold probe accuracy `0.9405`
- Size `1.75 cm`: `n=104`, out-of-fold probe accuracy `0.8942`

Interpretation boundary:
- This figure is more defensible than the global UMAP because it respects the known size-depth coupling.
- It still serves as representation evidence rather than a replacement for confusion matrices or balanced accuracy.