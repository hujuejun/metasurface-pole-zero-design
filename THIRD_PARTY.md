# Third-party attribution

The MIT license covers original code, not external libraries, publications, or third-party datasets.

- NumPy, SciPy, Matplotlib, and optional Tidy3D retain their respective licenses and service terms. They are installed as dependencies, not redistributed here.
- `inputs/ipdip/optical_constants.csv` is the numerical n,k table supplied for the manuscript's IP-Dip calculations (3–6 µm), retained verbatim. The manuscript cites Fullager et al., *Optical Materials Express* **7**, 888 (2017), DOI: [10.1364/OME.7.000888](https://doi.org/10.1364/OME.7.000888), for IP-Dip optical constants. The numerical checks do not independently authenticate the table's experimental provenance or uncertainty. The MIT code license does not purport to relicense the publication. `material_model.json` is the local fitted model.
- The idealized Sb2Se3 endpoint indices at 1550 nm are attributed to Delaney et al., *Advanced Functional Materials* (2020), DOI: [10.1002/adfm.202002447](https://doi.org/10.1002/adfm.202002447). The intermediate-state Lorentz–Lorenz model and limitations are recorded in the PCM material configuration.
- Corrected TM Fourier factorization follows Li (1996) and Lalanne and Morris (1996), cited in the manuscript SI. The historical design operator is not used for published verification.

No article PDFs, publisher figures, proprietary solver implementations, credentials, or cloud-result archives are included.
