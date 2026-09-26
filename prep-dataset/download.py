"""Retrieve SP L2/L2.1 and time-matched HMI observations for explicit IDs."""
import argparse
import csv
import json
from pathlib import Path
import requests


def ids_from_split(path):
    data = json.loads(path.read_text())
    return sorted({Path(x).stem for names in data.values() for x in names})


def download_sp(sample, level, output):
    from astropy.io import fits
    year, month, day = sample[:4], sample[4:6], sample[6:8]
    folder = 'level2hao' if level == '2' else 'level2.1hao'
    suffix = '.fits' if level == '2' else '_L2.1.fits'
    url = f'https://data.darts.isas.jaxa.jp/pub/hinode/sot/{folder}/{year}/{month}/{day}/SP3D/{sample}/{sample}{suffix}'
    target = output/(sample+'.fits')
    if target.exists():
        with fits.open(target): pass
        return url
    temporary = target.with_suffix('.fits.part')
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for block in response.iter_content(1024*1024): f.write(block)
        with fits.open(temporary) as hdus: hdus.verify('exception')
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()
    return url


def download_hmi(sample, output, email):
    from astropy.time import Time
    import astropy.units as u
    import numpy as np
    from sunpy.net import Fido, attrs as a
    stamp = Time(f'{sample[:4]}-{sample[4:6]}-{sample[6:8]}T{sample[9:11]}:{sample[11:13]}:{sample[13:15]}', scale='utc')
    query = Fido.search(a.Time(stamp-360*u.s, stamp+360*u.s),
                        a.jsoc.Series('hmi.M_720s'), a.jsoc.Segment('magnetogram'), a.jsoc.Notify(email))
    table = query[0]
    if not len(table): raise ValueError('No HMI record within 360 seconds')
    def parse_tai(value):
        value = str(value).replace('_TAI', '').replace('.', '-', 2).replace('_', 'T', 1)
        return Time(value, scale='tai')
    distances = [abs((parse_tai(row['T_REC'])-stamp).to_value(u.s)) for row in table]
    index = int(np.argmin(distances))
    # JSOC exports operate on query blocks, so re-query the selected record
    # instead of assuming a sliced response narrows the export.
    chosen = parse_tai(table[index]['T_REC'])
    exact = Fido.search(a.Time(chosen-1*u.s, chosen+1*u.s),
                        a.jsoc.Series('hmi.M_720s'), a.jsoc.Segment('magnetogram'), a.jsoc.Notify(email))
    result = Fido.fetch(exact, path=str(output), overwrite=False)
    if getattr(result, 'errors', []) or len(result) != 1:
        raise RuntimeError(f'Expected one successful HMI download, got {result}')
    return Path(result[0]).name, float(distances[index])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('instrument', choices=['sp', 'hmi'])
    p.add_argument('--split', type=Path, default=Path('data/splits/dataset_split_10115.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--level', choices=['2', '2.1'], default='2.1')
    p.add_argument('--email', help='Your registered JSOC email (HMI only)')
    p.add_argument('--limit', type=int, help='Download a small subset for a smoke test')
    a = p.parse_args()
    if a.instrument == 'hmi' and not a.email: p.error('--email is required for JSOC')
    a.output.mkdir(parents=True, exist_ok=True)
    samples = ids_from_split(a.split)
    if a.limit is not None: samples = samples[:a.limit]
    records, errors = [], []
    for sample in samples:
        try:
            if a.instrument == 'sp':
                records.append(dict(sample=sample, url=download_sp(sample, a.level, a.output)))
            else:
                file, dt = download_hmi(sample, a.output, a.email)
                records.append(dict(sample=sample, hmi_file=file, delta_seconds=dt))
            print(sample, flush=True)
        except Exception as e: errors.append(dict(sample=sample, reason=str(e)))
    if records:
        with (a.output/'pairs.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    (a.output/'download_errors.json').write_text(json.dumps(errors, indent=2))
    if errors: raise SystemExit(f'{len(errors)} downloads failed; see download_errors.json')


if __name__ == '__main__': main()
