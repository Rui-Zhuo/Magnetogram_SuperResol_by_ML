# Prepare the paired dataset

This folder contains HMI/SP download, coarse alignment, local registration, cropping and NPZ generation. After installing the repository, its Python module name is `prep_dataset`.

1. Obtain the updated SP pointing FITS files described in [data access](../docs/data_access.md) and place them in `data/raw/pointing/`, named by observation ID.
2. Download the paired raw data using your registered JSOC email:

```bash
python -m prep_dataset.download sp --level 2 --output data/raw/sp_l2
python -m prep_dataset.download sp --level 2.1 --output data/raw/sp_l21
python -m prep_dataset.download hmi --email YOUR_REGISTERED_JSOC_EMAIL --output data/raw/hmi
```

3. Align and prepare the paired patches:

```bash
python -m prep_dataset.coalign --pointing data/raw/pointing --sp-l2 data/raw/sp_l2 --sp-l21 data/raw/sp_l21 --hmi data/raw/hmi --pairs data/raw/hmi/pairs.csv --output data/aligned
python -m prep_dataset.prepare --input data/aligned --output data/paired --split data/splits/dataset_split_10115.json
```

Each paired NPZ contains `HMIfield` (200 × 200, G), `SPfield` (312 × 336, G) and `Txy` (200 × 200, projected radius in arcsec). The supplied split contains 6,069 training, 2,023 validation and 2,023 test samples. Use that split for the paper experiments. `split.py` creates a new split only when preparing a new dataset.

Display a downloaded magnetogram with `python -m evaluation.plot_magnetogram --input PATH_TO_HMI.fits --hdu 1 --output outputs/hmi.png`; SP L2.1 uses HDU 4.
