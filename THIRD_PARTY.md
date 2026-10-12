# Third-party attribution

The MIT license covers original code, not external libraries, publications, or third-party datasets.

- NumPy, SciPy, Matplotlib, and optional Tidy3D retain their respective licenses and service terms. They are installed as dependencies, not redistributed here.
- The numerical IP-Dip n,k table used for the manuscript's 3–6 µm calculations is **not redistributed**. The manuscript cites Fullager et al., *Optical Materials Express* **7**, 888 (2017), DOI: [10.1364/OME.7.000888](https://doi.org/10.1364/OME.7.000888), for IP-Dip optical constants; we have not independently authenticated the original numerical table's experimental provenance or uncertainty. Readers must supply their own appropriately licensed table for exact real-frequency replay or refitting. `inputs/ipdip/material_model.json` is the retained, approximate fitted model used for complex-frequency optimization; the MIT code license does not extend to third-party material measurements or publications.
- The idealized Sb2Se3 endpoint indices at 1550 nm are attributed to Delaney et al., *Advanced Functional Materials* (2020), DOI: [10.1002/adfm.202002447](https://doi.org/10.1002/adfm.202002447). The intermediate-state Lorentz–Lorenz model and limitations are recorded in the PCM material configuration.
- Corrected TM Fourier factorization follows Li (1996) and Lalanne and Morris (1996), cited in the manuscript SI. The historical design operator is not used for published verification.

No article PDFs, publisher figures, proprietary solver implementations, credentials, or cloud-result archives are included.
