# Sycophancy_AANP

A repository containing code for "Mitigating sycophancy in large language models via Sparse Activation Fusion and Multi-Layer Activation Steering".

This root README provides a high-level overview and points to the detailed implementation and documentation in the MLA directory.

## Quick links

- Detailed implementation and usage: MLA/README.md
- License: (see repository root for license file if present)

## Summary

The project implements Multi-Layer Activation Steering (MLAS) and related tooling to identify and remove sycophantic "pressure directions" from transformer activations. The full implementation, evaluation scripts, and instructions live in the MLA/ directory.

## Quick start

1. See MLA/README.md for installation and usage instructions.
2. Typical workflow:
   - Install dependencies: `pip install -r MLA/requirements.txt`
   - Configure your HuggingFace token as described in MLA/README.md
   - Run the main pipeline: `python MLA/main.py`

## Repository layout

- MLA/: Implementation, experiments, and detailed README

## Contact / Citation

If you use this code, please cite the accompanying paper (citation details will be added upon acceptance).

---

If you'd like, I can:
- Expand the root README with installation and example commands copied from MLA/README.md
- Add badges (CI, license, PyPI)
- Create a shorter abstract for the paper and add citation placeholder metadata

Tell me which of the above you'd like and I'll update the file accordingly.
