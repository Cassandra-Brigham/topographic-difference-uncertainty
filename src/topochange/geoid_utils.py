"""Geoid grid discovery and management utilities.

Provides functions for finding PROJ data directories, searching for geoid grids,
and resolving geoid model aliases to grid file paths.
"""
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyproj import datadir


# -------- PROJ Data Directory Detection ---------

def get_all_proj_data_dirs() -> List[str]:
    """
    Get all PROJ data directories that might be used by pyproj, PDAL, or GDAL.

    This function discovers directories from multiple sources:
    1. pyproj's configured data directory
    2. pyproj's user data directory
    3. PROJ_LIB and PROJ_DATA environment variables
    4. System PROJ directories (from `projinfo --searchpaths` if available)
    5. Common system paths (/usr/share/proj, /usr/local/share/proj, etc.)

    Returns
    -------
    List[str]
        List of existing directories where PROJ looks for grid files.
        Directories are deduplicated and only existing paths are returned.
    """
    dirs = set()

    # 1. pyproj's data directory
    try:
        dd = datadir.get_data_dir()
        if dd and os.path.isdir(dd):
            dirs.add(dd)
    except Exception:
        pass

    # 2. pyproj's user data directory
    try:
        ud = datadir.get_user_data_dir()
        if ud and os.path.isdir(ud):
            dirs.add(ud)
    except Exception:
        pass

    # 3. Environment variables
    for env_var in ['PROJ_LIB', 'PROJ_DATA']:
        env_val = os.environ.get(env_var, '')
        for path in env_val.split(os.pathsep):
            if path and os.path.isdir(path):
                dirs.add(path)

    # 4. System PROJ directories (via projinfo --searchpaths)
    try:
        result = subprocess.run(
            ['projinfo', '--searchpaths'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and os.path.isdir(line):
                    dirs.add(line)
    except Exception:
        pass

    # 5. Common system paths
    common_paths = [
        '/usr/share/proj',
        '/usr/local/share/proj',
        '/opt/conda/share/proj',
        '/opt/homebrew/share/proj',  # macOS Homebrew ARM
        '/usr/local/opt/proj/share/proj',  # macOS Homebrew Intel
    ]
    for path in common_paths:
        if os.path.isdir(path):
            dirs.add(path)

    return sorted(dirs)


def get_primary_proj_data_dir() -> str:
    """
    Get the primary PROJ data directory where grids should be downloaded.

    Priority:
    1. PROJ_LIB environment variable (if set and writable)
    2. First writable directory from projinfo --searchpaths
    3. pyproj's data directory
    4. First writable common system path

    Returns
    -------
    str
        Path to the primary PROJ data directory

    Raises
    ------
    RuntimeError
        If no writable PROJ data directory can be found
    """
    def is_writable(path: str) -> bool:
        try:
            test_file = os.path.join(path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception:
            return False

    # 1. PROJ_LIB if set and writable
    proj_lib = os.environ.get('PROJ_LIB', '')
    if proj_lib and os.path.isdir(proj_lib) and is_writable(proj_lib):
        return proj_lib

    # 2. Check all discovered directories for writability
    for d in get_all_proj_data_dirs():
        if is_writable(d):
            return d

    # 3. Fallback to pyproj data dir (may not be writable)
    try:
        dd = datadir.get_data_dir()
        if dd and os.path.isdir(dd):
            return dd
    except Exception:
        pass

    raise RuntimeError(
        "Could not find a writable PROJ data directory. "
        "Try setting the PROJ_LIB environment variable to a writable directory."
    )


def ensure_proj_grid(
    grid_filename: str,
    *,
    download_if_missing: bool = True,
    copy_to_all_dirs: bool = True,
    verbose: bool = True,
) -> str:
    """
    Ensure a PROJ grid file is available in the PROJ data directories.

    This function checks if a grid file exists in any PROJ data directory,
    and optionally downloads it from the PROJ CDN if missing. It can also
    copy the grid to all PROJ directories to ensure compatibility with
    different tools (pyproj, PDAL, GDAL, etc.).

    Parameters
    ----------
    grid_filename : str
        Name of the grid file (e.g., 'us_noaa_g2012bu0.tif')
    download_if_missing : bool, default True
        If True, download the grid from cdn.proj.org if not found locally
    copy_to_all_dirs : bool, default True
        If True, copy the grid to all PROJ data directories for maximum
        compatibility (useful in environments like Google Colab where
        pyproj and PDAL may use different directories)
    verbose : bool, default True
        If True, print status messages

    Returns
    -------
    str
        Full path to the grid file

    Raises
    ------
    FileNotFoundError
        If the grid is not found and download_if_missing is False
    RuntimeError
        If download fails

    Examples
    --------
    >>> # Ensure GEOID12B grid is available
    >>> path = ensure_proj_grid('us_noaa_g2012bu0.tif')
    >>> print(path)
    /usr/local/share/proj/us_noaa_g2012bu0.tif

    >>> # Use in a transformation workflow
    >>> ensure_proj_grid('us_noaa_g2018u0.tif')  # GEOID18
    >>> ensure_proj_grid('us_noaa_geoid09_conus.tif')  # GEOID09
    """
    # Normalize filename (just the basename)
    grid_basename = os.path.basename(grid_filename)

    # Get all PROJ directories
    all_dirs = get_all_proj_data_dirs()

    if verbose:
        print(f"Checking for grid: {grid_basename}")
        print(f"  PROJ directories: {all_dirs}")

    # Check if grid exists in any directory
    existing_path = None
    for d in all_dirs:
        candidate = os.path.join(d, grid_basename)
        if os.path.isfile(candidate):
            existing_path = candidate
            if verbose:
                print(f"  Found existing grid: {existing_path}")
            break

    # If not found, optionally download
    if existing_path is None:
        if not download_if_missing:
            raise FileNotFoundError(
                f"Grid file '{grid_basename}' not found in any PROJ directory: {all_dirs}"
            )

        # Download to primary directory
        primary_dir = get_primary_proj_data_dir()
        dest_path = os.path.join(primary_dir, grid_basename)
        cdn_url = f"https://cdn.proj.org/{grid_basename}"

        if verbose:
            print(f"  Downloading from {cdn_url}...")

        try:
            urllib.request.urlretrieve(cdn_url, dest_path)
            existing_path = dest_path
            if verbose:
                size = os.path.getsize(dest_path)
                print(f"  Downloaded: {size:,} bytes -> {dest_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download grid '{grid_basename}' from {cdn_url}: {e}"
            )

    # Optionally copy to all directories for maximum compatibility
    if copy_to_all_dirs and existing_path:
        import shutil
        for d in all_dirs:
            dest = os.path.join(d, grid_basename)
            if dest != existing_path and not os.path.isfile(dest):
                try:
                    shutil.copy(existing_path, dest)
                    if verbose:
                        print(f"  Copied to: {dest}")
                except PermissionError:
                    if verbose:
                        print(f"  Skipped (permission denied): {dest}")
                except Exception as e:
                    if verbose:
                        print(f"  Skipped (error: {e}): {dest}")

    return existing_path


def ensure_proj_grids_for_region(
    region: str = 'us_noaa',
    *,
    verbose: bool = True,
) -> List[str]:
    """
    Ensure all PROJ grids for a given region/source are available.

    This is a convenience function that downloads commonly used grids
    for a specific region or data source.

    Parameters
    ----------
    region : str
        Region identifier. Common values:
        - 'us_noaa': US NOAA geoids (GEOID03, GEOID09, GEOID12B, GEOID18)
        - 'us_nga': US NGA global geoids (EGM96, EGM2008)
        - 'ca_nrc': Canada NRC geoids
        - 'au_ga': Australia GA geoids
        - 'uk_os': UK Ordnance Survey geoids
        - 'nz_linz': New Zealand LINZ geoids
    verbose : bool, default True
        If True, print status messages

    Returns
    -------
    List[str]
        List of paths to the downloaded/verified grid files
    """
    # Define common grids per region
    REGION_GRIDS = {
        'us_noaa': [
            'us_noaa_g2012bu0.tif',   # GEOID12B CONUS
            'us_noaa_g2018u0.tif',    # GEOID18 CONUS
            'us_noaa_geoid09_conus.tif',  # GEOID09 CONUS
            'us_noaa_geoid06_conus.tif',  # GEOID06 CONUS
            'us_noaa_geoid03_conus.tif',  # GEOID03 CONUS
        ],
        'us_nga': [
            'us_nga_egm96_15.tif',    # EGM96
            'us_nga_egm08_25.tif',    # EGM2008
        ],
        'ca_nrc': [
            'ca_nrc_HT2_2010v70.tif',
        ],
        'au_ga': [
            'au_ga_AUSGeoid2020_20180201.tif',
            'au_ga_AUSGeoid09_V1.01.tif',
        ],
        'uk_os': [
            'uk_os_OSGM15_GB.tif',
        ],
        'nz_linz': [
            'nz_linz_nzgeoid2016.tif',
        ],
    }

    grids_to_download = REGION_GRIDS.get(region.lower(), [])
    if not grids_to_download:
        if verbose:
            print(f"Unknown region '{region}'. Known regions: {list(REGION_GRIDS.keys())}")
        return []

    if verbose:
        print(f"Ensuring grids for region '{region}'...")

    downloaded = []
    for grid in grids_to_download:
        try:
            path = ensure_proj_grid(grid, verbose=verbose)
            downloaded.append(path)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not ensure {grid}: {e}")

    return downloaded


# -------- Geoid helpers ---------
GEOID_HINTS = [
    # Common North America
    r"\bNAVD ?88\b",
    r"\bNGVD ?29\b",
    r"\bGEOID(?:03|06|09|12[A-Z]?|18)\b",
    # Global geoids
    r"\bEGM(?:96|2008)\b",
    r"\bEGM ?96\b",
    r"\bEGM ?2008\b",
    # Canada, UK, AUS, EU examples
    r"\bCGVD(?:28|2013)\b",
    r"\bMSL\b",
    r"\bOSGM\d+\b",
    r"\bAHD(?: ?(71|83))?\b",
    r"\bDVR90\b",
    r"\bEVRF(?:2000|2007|2019)\b",
    # Generic vertical hints
    r"\bgeoid(?:grid|model)?\b",
    r"\bvertical cs\b",
    r"\bverticalcrs\b",
]

# Vertical EPSG names often contain these tokens
VERT_NAME_HINTS = [
    "height",
    "vertical",
    "orthometric",
    "geoid",
    "msl",
]

# PROJ parameters that imply a geoid grid application
PROJ_VERTICAL_PARAMS = [r"\+geoidgrids?=", r"\+vunits?=", r"\+vto_meter="]

WKT_VERTICAL_TOKENS = [
    "VERT_CS",
    "VERTCRS",
    "VERTICALCRS",
    "COMPD_CS",
    "COMPOUNDCRS",
    "GEOGCRS",  # compound often mixes horiz+vert
]


def _iter_text_fields(d: Any, path: str = "") -> List[Tuple[str, str]]:
    """
    Walk a structure and return (path, text) pairs for any string-ish value.
    """
    out: List[Tuple[str, str]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            newp = f"{path}.{k}" if path else str(k)
            out.extend(_iter_text_fields(v, newp))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            newp = f"{path}[{i}]"
            out.extend(_iter_text_fields(v, newp))
    else:
        if isinstance(d, (str, bytes, bytearray)):
            try:
                text = d.decode("utf-8") if isinstance(d, (bytes, bytearray)) else str(d)
            except Exception:
                text = str(d)
            if text.strip():
                out.append((path, text))
    return out


def _search_patterns(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _from_json_vertical(srs_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to interpret vertical CRS info from PDAL-style SRS JSON.
    """
    # Direct vertical CRS object
    vert = srs_json.get("vertical") or {}
    if isinstance(vert, dict) and vert:
        name = vert.get("name") or ""
        code = (vert.get("id") or {}).get("code")
        auth = (vert.get("id") or {}).get("authority")
        if any(tok in name.lower() for tok in VERT_NAME_HINTS) or code or auth:
            return {
                "name": name or None,
                "authority": auth,
                "code": code,
                "source": "srs.json.vertical",
            }

    # Some PDAL exports keep everything under 'json' root; check 'name'
    name = srs_json.get("name") or ""
    if any(t in name.lower() for t in VERT_NAME_HINTS) and srs_json.get("type", "").lower().startswith("vertical"):
        code = (srs_json.get("id") or {}).get("code")
        auth = (srs_json.get("id") or {}).get("authority")
        return {"name": name, "authority": auth, "code": code, "source": "srs.json"}

    return None


def parse_geoid_info(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect PDAL/LAZ metadata for geoid / vertical datum info.

    Returns a dict:
    {
      "vertical_datum": str|None,
      "geoid_model": str|None,
      "epsg_vertical": str|None,
      "evidence": [ ... ],          # list of (field_path, snippet)
      "confidence": "high|medium|low|none",
      "note": str                   # explanation
    }
    """
    evidence: List[Tuple[str, str]] = []

    # 1) JSON SRS (structured)
    srs = meta.get("srs") or {}
    srs_json = srs.get("json") or {}
    if isinstance(srs_json, dict):
        j = _from_json_vertical(srs_json)
        if j:
            name = j.get("name")
            code = j.get("code")
            auth = j.get("authority")
            ev = f"{name or ''} ({auth}:{code})".strip()
            if ev:
                evidence.append(("srs.json.vertical", ev))
            return {
                "vertical_datum": name,
                "geoid_model": name if name and ("geoid" in name.lower() or "egm" in name.lower()) else None,
                "epsg_vertical": f"{auth}:{code}" if auth and code else None,
                "evidence": evidence,
                "confidence": "high",
                "note": "Vertical CRS found in structured SRS JSON.",
            }

    # 2) Collect all string fields for pattern search
    texts = _iter_text_fields(meta)

    # 2a) Look for explicit EPSG vertical codes or names in WKT/JSON blocks
    wkt_like_paths = [p for p, t in texts if any(tok in t.upper() for tok in WKT_VERTICAL_TOKENS)]
    for p, t in texts:
        if p in wkt_like_paths or any(tok in t.upper() for tok in WKT_VERTICAL_TOKENS):
            # Vertical CRS name
            m = re.search(r'(VERT(?:ICAL)?_?CRS|VERT_CS)\s*\[\s*"([^"]+)"', t, flags=re.IGNORECASE)
            if m:
                name = m.group(2)
                evidence.append((p, f'Vertical CRS name: "{name}"'))
                return {
                    "vertical_datum": name,
                    "geoid_model": name if ("geoid" in name.lower() or "egm" in name.lower() or "msl" in name.lower()) else None,
                    "epsg_vertical": None,
                    "evidence": evidence,
                    "confidence": "high",
                    "note": "Vertical CRS discovered in WKT.",
                }
            # AUTHORITY["EPSG","57xx"]
            m2 = re.search(r'AUTHORITY\["EPSG","(57\d{2})"\]', t, flags=re.IGNORECASE)
            if m2:
                code = m2.group(1)
                evidence.append((p, f"Vertical EPSG code: {code}"))
                return {
                    "vertical_datum": None,
                    "geoid_model": None,
                    "epsg_vertical": f"EPSG:{code}",
                    "evidence": evidence,
                    "confidence": "medium",
                    "note": "Found an EPSG code in a vertical section; resolve externally to identify datum/geoid.",
                }

    # 2b) PROJ strings: +geoidgrids= or vertical params
    for p, t in texts:
        if "proj4" in p.lower() or "+proj=" in t:
            ev = _search_patterns(t, PROJ_VERTICAL_PARAMS)
            if ev:
                evidence.append((p, f"PROJ parameter indicates vertical grid: {ev}"))
                grid = None
                mg = re.search(r"\+geoidgrids?=([^ \t]+)", t)
                if mg:
                    grid = mg.group(1)
                    evidence.append((p, f"Grid: {grid}"))
                return {
                    "vertical_datum": None,
                    "geoid_model": grid,
                    "epsg_vertical": None,
                    "evidence": evidence,
                    "confidence": "medium",
                    "note": "PROJ string references a geoid grid.",
                }

    # 2c) GeoTIFF keys: look for VerticalCSTypeGeoKey / citations
    gtiff = meta.get("gtiff") or ""
    if isinstance(gtiff, str) and gtiff:
        if "VerticalCSTypeGeoKey" in gtiff or re.search(r"Vertical.*GeoKey", gtiff, re.IGNORECASE):
            evidence.append(("gtiff", "VerticalCSTypeGeoKey present"))
            m = re.search(r'GeogCitationGeoKey\s*\(Ascii,\d+\):\s*"([^"]+)"', gtiff)
            if m:
                evidence.append(("gtiff", f'Citation: "{m.group(1)}"'))
            return {
                "vertical_datum": None,
                "geoid_model": None,
                "epsg_vertical": None,
                "evidence": evidence,
                "confidence": "low",
                "note": "GeoTIFF reports vertical keys, but no explicit model parsed. Inspect full keys.",
            }

    # 2d) Generic token search anywhere
    for p, t in texts:
        token = _search_patterns(t, GEOID_HINTS)
        if token:
            evidence.append((p, f"Token: {token}"))
            return {
                "vertical_datum": token,
                "geoid_model": token if token.upper().startswith(("GEOID", "EGM")) else None,
                "epsg_vertical": None,
                "evidence": evidence,
                "confidence": "medium",
                "note": "Found a recognizable geoid/vertical-datum token.",
            }

    # 3) PDAL convenience flags implying 'no vertical CRS'
    units_vertical = ((meta.get("srs") or {}).get("units") or {}).get("vertical", "")
    srs_vertical = (meta.get("srs") or {}).get("vertical", "")
    if (not srs_vertical) and (not units_vertical):
        evidence.append(("srs.units.vertical", repr(units_vertical)))
        evidence.append(("srs.vertical", repr(srs_vertical)))
        note = (
            "No vertical CRS or geoid model found. Metadata indicates only a horizontal CRS "
            "(e.g., UTM on WGS84) with empty vertical fields; z values are likely ellipsoidal heights."
        )
        return {
            "vertical_datum": None,
            "geoid_model": None,
            "epsg_vertical": None,
            "evidence": evidence,
            "confidence": "none",
            "note": note,
        }

    # Fallback: unknown
    return {
        "vertical_datum": None,
        "geoid_model": None,
        "epsg_vertical": None,
        "evidence": evidence,
        "confidence": "low",
        "note": "No explicit vertical datum/geoid tokens detected. Manual verification recommended.",
    }


def list_proj_geoid_grids(
    include_cdn: bool = True,
    merge: bool = True,
    as_paths: bool = True,
) -> Dict[str, List[str]]:
    """
    List all geoid grid files accessible to PROJ â€” locally and optionally from the CDN.
    """
    # Local search
    dirs = set()
    dd = datadir.get_data_dir()
    if dd and os.path.isdir(dd):
        dirs.add(dd)
    try:
        ud = datadir.get_user_data_dir()
        if ud and os.path.isdir(ud):
            dirs.add(ud)
    except Exception:
        pass

    for env in os.environ.get("PROJ_LIB", "").split(os.pathsep):
        if env and os.path.isdir(env):
            dirs.add(env)

    geoid_hint = re.compile(r"(geoid|g|egm|osgm|cgg|geoide|hbg|swen|geodpt|rh2000)", re.IGNORECASE)
    patterns = (".tif", ".gtx", ".byn", ".bin")
    local_files: List[str] = []

    for base in dirs:
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(patterns) and geoid_hint.search(f):
                    local_files.append(os.path.join(root, f))

    local_files = sorted(local_files)
    local_display = [Path(f).name if not as_paths else f for f in local_files]

    # CDN search
    cdn_files: List[str] = []
    if include_cdn:
        cdn_url = "https://cdn.proj.org/"
        cdn_hints = (
            "GEOID",
            "EGM",
            "us_noaa",
            "us_nga",
            "ca_nrc",
            "HT2",
            "CGG",
            "uk_os",
            "OSGM",
            "be_ign",
            "HBG",
            "ar_ign",
            "GEOIDE-AR",
            "at_bev",
            "BESSEL",
            "GRS80",
            "GV_Hoehengrid",
            "au_ga",
            "AUSGeoid",
            "es_ign",
            "rednap",
            "is_lmi",
            "Icegeoid",
            "ISN",
            "hu_bme",
            "geoid2014",
            "nz_linz",
            "nzgeoid",
            "pl_gugik",
            "EVRF2007",
            "KRON86",
            "pt_dgt",
            "GeodPT",
            "se_lantmateriet",
            "SWEN",
            "RH2000",
            "za_cdngi",
            "sageoid",
        )
        try:
            with urllib.request.urlopen(cdn_url, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            all_tifs = re.findall(r'href="([A-Za-z0-9_\-\.]+\.tif)"', html)
            cdn_files = sorted({f for f in all_tifs if any(h.lower() in f.lower() for h in cdn_hints)})
        except Exception:
            cdn_files = []

    cdn_display = sorted(cdn_files)

    # Merge
    merged = []
    if merge:
        local_basenames = {Path(f).name for f in local_files}
        merged = list(local_display)
        for f in cdn_display:
            if f not in local_basenames:
                merged.append(f)
        merged = sorted(merged)

    result: Dict[str, List[str]] = {"local": local_display, "cdn": cdn_display}
    if merge:
        result["all"] = merged
    return result


# ---------- Normalization & Aliasing ----------


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\.(tif|gtx|byn|bin)$", "", s)
    s = re.sub(r"[\s\-_]", "", s)
    replacements = {
        "earthgravitymodel": "egm",
        "geoidmodel": "geoid",
        "model": "",
        "grid": "",
        "v": "",
        "ver": "",
        "version": "",
        "rev": "",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r"20(\d{2})([a-z]?)", r"\1\2", s)
    s = re.sub(
        r"^(us|ca|uk|gb|au|nz|es|pt|pl|se|za|ar|is|at)"
        r"(noaa|nga|nrc|os|ign|bev|ga|linz|dgt|gugik|lantmateriet|cdngi)?",
        "",
    )
    s = re.sub(r"(datum|grs80|bessel)$", "", s)
    return s


# ---------- Alias definitions ----------

ALIAS_GROUPS = {
    "egm2008": ("egm2008", "egm08"),
    "egm96": ("egm96", "egm1996"),
    "geoid18": ("geoid 18", "geoid18", "g18", "g2018", "geoid_2018", "geoid18u", "18conus"),
    "geoid12b": ("geoid12b", "geoid12", "g12b", "g2012b", "g12", "g2012", "12conus", "geoid 12b", "geoid 12"),
    "geoid12a": ("geoid12a", "g12a", "g2012a", "geoid 12a"),
    "geoid09": ("geoid09", "geoid9", "g9", "g09", "g2009", "geoid 09", "geoid 9"),
    "geoid03": ("geoid03", "geoid3", "g3", "g03", "g2003", "geoid 03", "geoid 3"),
    "osgm15": ("osgm15", "osgm36", "osgm02"),
    "ht2": ("ht2", "ht20"),
    "cgg2013a": ("cgg2013a", "cgg2013", "cgg2010", "cgg"),
    "ausgeoid2020": ("ausgeoid2020", "ausgeoid09", "ausgeoid98", "ausgeoid"),
    "nzgeoid2016": ("nzgeoid2016", "nzgeoid2009", "nzgeoid"),
    "rednap": ("rednap",),
    "geodpt": ("geodpt2011", "geodpt08", "geodpt"),
    "swen17": ("swen17", "swen"),
    "icegeoid": ("icegeoid", "isn93", "isn2004", "isn2016"),
    "sageoid": ("sageoid",),
    "geoidear": ("geoidear",),
    "gvhoehengrid": ("gvhoehengrid", "hoehen", "hoehengrid"),
}

grids = list_proj_geoid_grids(include_cdn=True, merge=True, as_paths=True)
_local_basenames = {Path(f).name.lower() for f in grids["local"]}


def _matches_alias(aliases: Tuple[str, ...], filename: str) -> bool:
    stem = Path(filename).stem.lower()
    for alias in aliases:
        a = alias.lower()
        L = len(a)
        has_digit = any(ch.isdigit() for ch in a)
        if L <= 3:
            pattern = rf"(^|[_\-.]){re.escape(a)}($|[_\-.])"
            if re.search(pattern, stem):
                return True
        elif has_digit:
            pattern = rf"(^|[_\-.]){re.escape(a)}"
            if re.search(pattern, stem):
                return True
        else:
            if a in stem:
                return True
    return False


alias_to_files: Dict[str, List[str]] = {}
canonical_from_alias: Dict[str, str] = {}
for canonical, aliases in ALIAS_GROUPS.items():
    for alias in aliases:
        alias_lower = alias.lower()
        canonical_from_alias[alias_lower] = canonical
        matched = [f for f in grids["all"] if _matches_alias(aliases, f)]
        seen = set()
        cleaned: List[str] = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                cleaned.append(m)
        alias_to_files[alias_lower] = cleaned


def _is_local(f: str) -> bool:
    if f in grids["local"]:
        return True
    return Path(f).name.lower() in _local_basenames


def _is_conus(f: str) -> bool:
    """
    Generic CONUS detection:
      - Any filename containing 'u0'
      - Any filename containing 'conus'
    """
    stem = Path(f).stem.lower()
    return "u0" in stem or "conus" in stem


def select_geoid_grid(
    name: str,
    *,
    choice: int | None = None,
    verbose: bool = True,
    ensure_available: bool = True,
) -> Tuple[str, List[str]]:
    """
    Select a geoid grid for the given nickname/alias.

    Rules (applied to all models):
      1. Local CONUS-style grid (u0 / conus)
      2. Any CONUS-style grid
      3. Other local grid
      4. First available grid (downloads if ensure_available=True)

    Also handles direct filenames - if name looks like a filename (ends with
    .tif, .gtx, etc.), it will be returned directly if the file exists, or
    the alias will be extracted from the filename pattern.

    Parameters
    ----------
    name : str
        Geoid alias (e.g., 'geoid12b', 'geoid18', 'egm2008') or filename
    choice : int, optional
        Override auto-selection with a specific index from candidates
    verbose : bool, default True
        Print status messages
    ensure_available : bool, default True
        If True and the selected grid is not locally available, automatically
        download it from cdn.proj.org. This ensures the grid is ready for use
        by PDAL, pyproj, and other PROJ-based tools.

    Returns
    -------
    Tuple[str, List[str]]
        (selected_grid_path, list_of_all_candidates)
    """
    name_lower = name.lower()
    
    # Check if input is a filename rather than an alias
    geoid_extensions = ('.tif', '.gtx', '.gtx.gz', '.gvb', '.byn', '.grid')
    if any(name_lower.endswith(ext) for ext in geoid_extensions):
        # It's a filename - check if it exists directly
        if os.path.isfile(name):
            if verbose:
                print(f"Geoid grid (direct file): {name}")
            return name, [name]
        
        # Check in PROJ data directory
        proj_dir = datadir.get_data_dir()
        if proj_dir:
            proj_path = os.path.join(proj_dir, os.path.basename(name))
            if os.path.isfile(proj_path):
                if verbose:
                    print(f"Geoid grid (PROJ dir): {proj_path}")
                return proj_path, [proj_path]
        
        # Try to extract alias from filename pattern
        # e.g., "us_noaa_geoid09_conus.tif" -> "geoid09"
        # e.g., "us_noaa_g2018u0.tif" -> "geoid18"
        basename = os.path.basename(name_lower)
        
        # Pattern: geoidXX or gXXXX
        match = re.search(r'geoid(\d+)', basename)
        if match:
            extracted_alias = f"geoid{match.group(1)}"
            if extracted_alias in alias_to_files:
                name_lower = extracted_alias
                if verbose:
                    print(f"Extracted alias '{extracted_alias}' from filename '{name}'")
        else:
            # Try pattern like g2018, g2012
            match = re.search(r'g(\d{4})', basename)
            if match:
                year = match.group(1)
                # Map year to geoid version
                year_to_alias = {
                    '2003': 'geoid03',
                    '2006': 'geoid06',
                    '2009': 'geoid09',
                    '2012': 'geoid12b',
                    '2018': 'geoid18',
                }
                if year in year_to_alias:
                    extracted_alias = year_to_alias[year]
                    if extracted_alias in alias_to_files:
                        name_lower = extracted_alias
                        if verbose:
                            print(f"Extracted alias '{extracted_alias}' from filename '{name}'")
    
    alias = name_lower
    if alias not in alias_to_files:
        raise ValueError(f"No known alias group for '{name}'")
    candidates = alias_to_files[alias]
    if not candidates:
        raise ValueError(f"No grid files found for alias '{name}'")

    canonical = canonical_from_alias.get(alias, alias)

    if choice is not None:
        if not (0 <= choice < len(candidates)):
            raise IndexError(f"choice={choice} out of range for '{name}'")
        selected = candidates[choice]
        reason = "user override"
    else:
        conus_local = [f for f in candidates if _is_local(f) and _is_conus(f)]
        conus_any = [f for f in candidates if _is_conus(f)]
        non_conus_local = [f for f in candidates if _is_local(f) and not _is_conus(f)]
        if conus_local:
            selected = conus_local[0]
            reason = "auto-selected local u0/CONUS grid"
        elif conus_any:
            selected = conus_any[0]
            reason = "auto-selected u0/CONUS grid (non-local)"
        elif non_conus_local:
            selected = non_conus_local[0]
            reason = "auto-selected local grid"
        else:
            selected = candidates[0]
            reason = "auto-selected first available grid"

    if verbose:
        print(f"Geoid grids for '{name}' (canonical '{canonical}'):")
        for i, f in enumerate(candidates):
            base = Path(f).name
            origin = "local" if _is_local(f) else "CDN"
            mark = f"  <== selected ({reason})" if f == selected else ""
            print(f"  [{i}] {base} ({origin}){mark}")

    # If the selected grid is not local and ensure_available is True, download it
    if ensure_available and not _is_local(selected):
        grid_basename = Path(selected).name
        if verbose:
            print(f"\nSelected grid '{grid_basename}' is not locally available.")
            print("Downloading and installing to PROJ data directories...")
        try:
            local_path = ensure_proj_grid(
                grid_basename,
                download_if_missing=True,
                copy_to_all_dirs=True,
                verbose=verbose,
            )
            # Update selected to point to the local path
            selected = local_path
            if verbose:
                print(f"Grid now available at: {selected}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not download grid: {e}")
                print("You may need to download it manually from https://cdn.proj.org/")

    return selected, candidates