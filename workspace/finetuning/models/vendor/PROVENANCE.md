# Architecture provenance

`lwnet.py` is copied without changes from
`LWNet/training/models/models_lw_3D.py`, submodule commit
`ca8361cd4a1c6418e3366fcc2ebc91fb72ac8c78`, upstream
https://github.com/hkzhang-git/LWNet (TGRS 2019, "Hyperspectral classification
based on lightweight 3-D-CNN with transfer learning"). No LICENSE/COPYING/NOTICE
file or license grant was found in the supplied checkout. No license is inferred
or granted here for this upstream source; redistribution rights need clarification
before publishing a package containing it.

`n2n4m.py` is copied without changes from `n2n4m-main/n2n4m/model.py`, upstream
https://github.com/rob-platt/N2N4M. The supplied checkout has no recorded commit
here. Its MIT license (Copyright (c) 2024 rob-platt) is preserved verbatim in
`N2N4M_LICENSE`.

The wrappers change the computational path (segmentation and padding). Loading
original weights is initialization for fine-tuning, not an assertion of inference
equivalence to the original pipeline.
