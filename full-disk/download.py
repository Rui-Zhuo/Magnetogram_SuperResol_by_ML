"""Download the HMI record and optional SP comparison observation for a paper case."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True, help='examples/application/CASE/case.json')
    p.add_argument('--email', required=True, help='Your registered JSOC email')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--with-sp', action='store_true', help='Also download the SP comparison map')
    a = p.parse_args()
    from astropy.time import Time
    import astropy.units as u
    from astropy.io import fits
    from sunpy.net import Fido, attrs
    from prep_dataset.download import download_sp
    case = json.loads(a.case.read_text(encoding='utf-8'))
    stamp = case['hmi_record'].replace('_TAI', '').replace('.', '-', 2).replace('_', 'T', 1)
    time = Time(stamp, scale='tai')
    a.output.mkdir(parents=True, exist_ok=True)
    raw = a.output/'downloads'
    raw.mkdir(exist_ok=True)
    query = Fido.search(attrs.Time(time-1*u.s, time+1*u.s),
                        attrs.jsoc.Series('hmi.M_720s'), attrs.jsoc.Segment('magnetogram'),
                        attrs.jsoc.Notify(a.email))
    if len(query) != 1 or len(query[0]) != 1:
        raise RuntimeError('Expected exactly one HMI record for the requested case')
    result = Fido.fetch(query, path=str(raw), overwrite=False)
    if getattr(result, 'errors', []) or len(result) != 1:
        raise RuntimeError('HMI export failed or returned an unexpected number of files')
    with fits.open(result[0]) as hdus:
        hdus.verify('exception')
    shutil.copyfile(result[0], a.output/'hmi.fits')
    if a.with_sp:
        download_sp(case['sp_observation'], '2.1', raw)
        shutil.copyfile(raw/(case['sp_observation']+'.fits'), a.output/'sp.fits')
    files = ['hmi.fits', 'sp.fits'] if a.with_sp else ['hmi.fits']
    log = dict(case=case['case'], hmi_record=case['hmi_record'],
               files={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in files})
    (a.output/'download.json').write_text(json.dumps(log, indent=2)+'\n', encoding='utf-8')
    print(a.output/'hmi.fits')


if __name__ == '__main__':
    main()
