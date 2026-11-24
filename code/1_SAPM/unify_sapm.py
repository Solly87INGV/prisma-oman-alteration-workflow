#!/usr/bin/env python3
"""Unisce i tre GeoTIFF della SAPM (Fig. 3 e 5) in un'unica maschera binaria 0/255.

Uso:
    python3 unify_sapm.py <out.tif> <in1.tif> <in2.tif> [<in3.tif> ...]

- Griglia di uscita: CRS e risoluzione del primo file, estensione = unione di tutti.
- Ricampionamento nearest (nessuna interpolazione dei valori).
- Target = valore > 0 e diverso dal nodata in almeno uno dei file; tutto il resto = 0.
- Si ferma se un file contiene piu' di due valori distinti (non e' una maschera).
"""
import sys
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

out, inputs = sys.argv[1], sys.argv[2:]
if len(inputs) < 2:
    sys.exit("Servono almeno due file in ingresso.")

ref = gdal.Open(inputs[0])
srs = ref.GetProjection()
gt = ref.GetGeoTransform()
xres, yres = gt[1], abs(gt[5])

# 1) controllo dei valori e porto tutto nel CRS/risoluzione del primo file
vrts, bounds = [], []
for f in inputs:
    ds = gdal.Open(f)
    b = ds.GetRasterBand(1)
    nd = b.GetNoDataValue()
    a = b.ReadAsArray()
    vals = np.unique(a[~np.isnan(a)] if a.dtype.kind == "f" else a)
    if nd is not None:
        vals = vals[vals != nd]
    print(f"\n{f}\n  dimensioni {ds.RasterXSize}x{ds.RasterYSize}, nodata={nd}, "
          f"valori={vals[:10]}{' ...' if vals.size > 10 else ''}")
    if vals.size > 2:
        sys.exit("  -> piu' di due valori distinti: non e' una maschera binaria, mi fermo.")
    v = gdal.Warp("", ds, format="VRT", dstSRS=srs, xRes=xres, yRes=yres,
                  targetAlignedPixels=True, resampleAlg="near")
    g = v.GetGeoTransform()
    bounds.append((g[0], g[3] + g[5] * v.RasterYSize, g[0] + g[1] * v.RasterXSize, g[3]))
    vrts.append((v, nd))

ob = (min(b[0] for b in bounds), min(b[1] for b in bounds),
      max(b[2] for b in bounds), max(b[3] for b in bounds))

# 2) unione sulla griglia comune
union, counts = None, []
for v, nd in vrts:
    w = gdal.Warp("", v, format="MEM", outputBounds=ob, xRes=xres, yRes=yres,
                  resampleAlg="near", dstNodata=np.nan, outputType=gdal.GDT_Float32)
    a = w.GetRasterBand(1).ReadAsArray()
    m = np.isfinite(a) & (a > 0)
    if nd is not None:
        m &= a != nd
    counts.append(int(m.sum()))
    union = m if union is None else (union | m)

# 3) scrittura 0/255, Byte, senza nodata
drv = gdal.GetDriverByName("GTiff")
ny, nx = union.shape
o = drv.Create(out, nx, ny, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "TILED=YES"])
o.SetGeoTransform((ob[0], xres, 0, ob[3], 0, -yres))
o.SetProjection(srs)
o.GetRasterBand(1).WriteArray(np.where(union, 255, 0).astype(np.uint8))
o.GetRasterBand(1).SetDescription("SAPM (0 = non-target, 255 = target)")
o = None

print(f"\nPixel target per file: {counts}")
print(f"Pixel target nell'unione: {int(union.sum())} (sovrapposti: {sum(counts) - int(union.sum())})")
print(f"Scritto: {out}  ({nx}x{ny} pixel, risoluzione {xres} x {yres})")
