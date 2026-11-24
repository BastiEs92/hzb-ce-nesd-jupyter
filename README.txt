XRD Rietveld — test folder
==========================
Contents:
  refine.ipynb          the tool UI (open this)
  rietveld_batch.py     refinement engine
  bruker_raw.py         Bruker .raw reader (fallback)
  Base.xy               TEST PATTERN  (reference HEO spinel, 500 C)
  magnetite.cif         TEST PHASE    (Fd-3m spinel template)
  default_Cu_lab.instprm Cu K-alpha instrument parameters

How to use in NOMAD JupyterLab:
  1. Upload this whole folder into /work (or upload the .zip and run
     `unzip xrd_rietveld_test.zip` in a Terminal).
  2. Open refine.ipynb. Run the cell.
  3. Data folder = this folder ("." usually works). Click "Scan folder".
  4. Pattern = Base.xy ; Phase CIF = magnetite.cif ; Instrument = default_Cu_lab.instprm
  5. Click "Run refinement".

NOTE: the refinement itself needs GSAS-II in the kernel. Until GSAS-II is
added to the NORTH base image, the app still opens and scans files, but the
"Run refinement" button shows a "GSAS-II not found" banner. Once the image
has GSAS-II (import GSASII.GSASIIscriptable), this same folder runs for real.
