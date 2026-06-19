"""
utils.py — shared functions for the QSAR selectivity interpretability project.
Imported by all four notebooks. Edit here; changes propagate everywhere.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (Descriptors, rdMolDescriptors,
                         rdFingerprintGenerator)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SaltRemover import SaltRemover
import json, os, tempfile

# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 42
N_FP       = 2048          # Morgan fingerprint size
N_PHYS     = 9             # number of physicochemical descriptors
RADIUS     = 2             # Morgan radius (ECFP4)
# assay_types is now a parameter to fetch_curate (default: ['Ki'])

_remover    = SaltRemover()
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_FP)


# ── ChEMBL helpers ─────────────────────────────────────────────────────────────
def fetch_curate(name: str, chembl_id: str,
                  assay_types: list = None) -> pd.DataFrame:
    """
    Fetch activity data from ChEMBL and curate:
    - Filter to specified assay types in nM (default: Ki only)
    - Convert to pChEMBL = 9 - log10(nM)
    - Strip salts, canonicalize SMILES
    - Deduplicate by median pChEMBL per unique structure
    Returns DataFrame with columns: curated_smiles, pChEMBL, molecule_chembl_id

    assay_types: list of standard_type values to include.
                 Default ['Ki'] for selectivity modelling.
                 Pass ['Ki','IC50','EC50','Potency','AC50'] for all types.
    """
    if assay_types is None:
        assay_types = ['Ki']
    df = _fetch_chembl_activities(chembl_id, assay_types)
    if df.empty:
        print(f"  {name}: no data returned")
        return pd.DataFrame()

    df = df.dropna(subset=['standard_value','canonical_smiles'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_value'] > 0].dropna(subset=['standard_value'])
    df['pChEMBL'] = 9 - np.log10(df['standard_value'])
    df = df[df['pChEMBL'] > 0]
    df['curated_smiles'] = df['canonical_smiles'].apply(_curate_smiles)
    df = df.dropna(subset=['curated_smiles'])

    agg = df.groupby('curated_smiles', as_index=False).agg(
        pChEMBL=('pChEMBL','median'),
        molecule_chembl_id=('molecule_chembl_id','first'))
    agg['receptor'] = name
    print(f"  {name} ({chembl_id}): {len(agg):,} unique structures")
    return agg


CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
CHEMBL_ACTIVITY_FIELDS = "activity_id,molecule_chembl_id,canonical_smiles,standard_value"


def _fetch_chembl_activities(chembl_id: str, assay_types: list) -> pd.DataFrame:
    """Pull activity rows from ChEMBL (Ki/IC50/… in nM).

    Uses direct REST pagination (limit=1000) instead of chembl_webresource_client,
    which defaults to 20 rows/page and often hits HTTP 500 on long downloads.
    """
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    page_size = 1000
    rows = []

    for atype in assay_types:
        offset = 0
        total = None
        while total is None or offset < total:
            params = {
                'target_chembl_id': chembl_id,
                'standard_type': atype,
                'standard_units': 'nM',
                'limit': page_size,
                'offset': offset,
                'only': CHEMBL_ACTIVITY_FIELDS,
            }
            url = CHEMBL_ACTIVITY_URL + '?' + urllib.parse.urlencode(params)
            last_err = None
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(url, timeout=120) as resp:
                        data = json.loads(resp.read().decode())
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                    last_err = err
                    if attempt == 5:
                        raise RuntimeError(
                            f"ChEMBL fetch failed ({chembl_id}, {atype}, offset={offset}): {err}"
                        ) from err
                    wait = 2 ** attempt
                    print(f"  ChEMBL page error ({chembl_id}, {atype}, offset={offset}), retry in {wait}s...")
                    time.sleep(wait)
            else:
                if last_err:
                    raise last_err

            meta = data.get('page_meta', {})
            total = meta.get('total_count', total or 0)
            batch = data.get('activities', [])
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < page_size:
                break

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if len(assay_types) > 1 and 'activity_id' in df.columns:
        df = df.drop_duplicates(subset=['activity_id'], keep='first')
    return df


def _curate_smiles(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(_remover.StripMol(mol), canonical=True)
    except Exception:
        return None


# ── Molecular representations ──────────────────────────────────────────────────
def get_fp(smi: str):
    """Return ECFP4 fingerprint as numpy int8 array, or None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp  = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros((N_FP,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_scaffold(smi: str) -> str:
    """Return Murcko scaffold SMILES, empty string on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ''


def get_physchem(smi: str):
    """Return 9 physicochemical descriptors as a list."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [0.0] * N_PHYS
    return [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
    ]


# ── Feature engineering ────────────────────────────────────────────────────────
def build_feature_matrix(fps_array: np.ndarray,
                          smiles_list: list,
                          selector,
                          scaler) -> np.ndarray:
    """
    Apply a (fitted) VarianceThreshold selector and StandardScaler to
    raw fingerprints + physicochemical descriptors.
    Returns float32 array of shape (n, n_kept_fp + N_PHYS).
    """
    fps  = fps_array.astype(np.float32)
    phys = np.array([get_physchem(s) for s in smiles_list], dtype=np.float32)
    phys = np.nan_to_num(phys, nan=0.0)
    fps_f  = selector.transform(fps)
    phys_s = scaler.transform(phys)
    return np.hstack([fps_f, phys_s]).astype(np.float32)


def fit_feature_pipeline(fps_array: np.ndarray,
                          smiles_list: list,
                          variance_threshold: float = 0.01):
    """
    Fit VarianceThreshold + StandardScaler on raw FP + physchem.
    Returns (X, selector, scaler).
    """
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    fps  = fps_array.astype(np.float32)
    phys = np.array([get_physchem(s) for s in smiles_list], dtype=np.float32)
    phys = np.nan_to_num(phys, nan=0.0)

    selector = VarianceThreshold(threshold=variance_threshold)
    fps_f    = selector.fit_transform(fps)
    scaler   = StandardScaler()
    phys_s   = scaler.fit_transform(phys)
    X = np.hstack([fps_f, phys_s]).astype(np.float32)
    return X, selector, scaler


# ── Scaffold splitting ─────────────────────────────────────────────────────────
def scaffold_split(df: pd.DataFrame,
                   test_frac: float = 0.2,
                   seed: int = SEED):
    """
    Split by unique Murcko scaffolds.
    Returns (train_indices, test_indices) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    scaffolds = df['scaffold'].values
    unique_sc = list(set(scaffolds))
    rng.shuffle(unique_sc)

    n_test_target = int(len(df) * test_frac)
    test_sc = set()
    n_test  = 0
    for sc in unique_sc:
        if n_test >= n_test_target:
            break
        test_sc.add(sc)
        n_test += (scaffolds == sc).sum()

    test_idx  = np.where(np.isin(scaffolds, list(test_sc)))[0]
    train_idx = np.where(~np.isin(scaffolds, list(test_sc)))[0]
    return train_idx, test_idx


def scaffold_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED):
    """
    Yield (train_idx, val_idx) tuples using scaffold-aware k-fold grouping.
    """
    rng = np.random.RandomState(seed)
    scaffolds = df['scaffold'].values
    unique_sc = list(set(scaffolds))
    rng.shuffle(unique_sc)

    folds      = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for sc in unique_sc:
        idx     = np.where(scaffolds == sc)[0].tolist()
        smallest = int(np.argmin(fold_sizes))
        folds[smallest].extend(idx)
        fold_sizes[smallest] += len(idx)

    for k in range(n_splits):
        val   = np.array(folds[k])
        train = np.array([i for j, f in enumerate(folds) if j != k for i in f])
        yield train, val


# ── SHAP utilities ─────────────────────────────────────────────────────────────
def patch_xgb(model):
    """
    Fix XGBoost 2.0+ base_score bracketed format ([value] -> value).
    Uses native JSON round-trip — more reliable than save_config/load_config.
    """
    if not hasattr(model, 'save_model'):
        return model
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    tmp.close()
    try:
        model.save_model(tmp.name)
        with open(tmp.name) as f:
            mj = json.load(f)
        bs = mj['learner']['learner_model_param']['base_score']
        if isinstance(bs, str) and bs.startswith('['):
            mj['learner']['learner_model_param']['base_score'] = bs[1:-1]
            with open(tmp.name, 'w') as f:
                json.dump(mj, f)
            model.load_model(tmp.name)
    except Exception as e:
        print(f"  XGBoost patch warning: {e}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return model


def expand_shap_to_fp(sv: np.ndarray,
                       selector,
                       n_fp: int = N_FP) -> np.ndarray:
    """
    Expand SHAP values from model-specific feature space back to common
    N_FP-dimensional fingerprint space (drops descriptor columns).
    Fills zeros at positions filtered out by the VarianceThreshold selector.
    Returns float32 array of shape (n_samples, n_fp).
    """
    n_kept  = int(selector.get_support().sum())
    sv_fp   = sv[:, :n_kept]
    sv_full = np.zeros((sv.shape[0], n_fp), dtype=np.float32)
    sv_full[:, selector.get_support()] = sv_fp
    return sv_full


def cv_r2(model, X: np.ndarray, y: np.ndarray,
          df_for_folds: pd.DataFrame, n_splits: int = 5) -> tuple:
    """
    Cross-validate using scaffold-aware folds.
    Returns (mean_r2, std_r2).
    """
    from sklearn.metrics import r2_score
    scores = []
    for tr, va in scaffold_kfold(df_for_folds, n_splits=n_splits, seed=SEED):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[va], m.predict(X[va])))
    return float(np.mean(scores)), float(np.std(scores))


# ── Fragment visualisation ────────────────────────────────────────────────────
def get_bit_fragment_img(smiles_list: list, bit_id: int, size=(200, 200)):
    """
    Return PIL Image of the Morgan bit fragment from the first molecule
    in smiles_list that contains bit_id. Returns None if not found.
    """
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image as PILImage
        import io
    except ImportError:
        return None

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bi = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, RADIUS, nBits=N_FP, bitInfo=bi)
        if bit_id not in bi:
            continue
        atom_idx, radius = bi[bit_id][0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        if not env:
            continue
        amap  = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(submol, highlightAtoms=list(amap.values()))
        drawer.FinishDrawing()
        return PILImage.open(io.BytesIO(drawer.GetDrawingText())).copy()
    return None


def draw_fragment_grid(bit_list: list,
                       importance_vec: np.ndarray,
                       smiles_pool: list,
                       title: str,
                       n: int = 12,
                       cols: int = 6,
                       save_path: str = None):
    """
    Draw a grid of molecular fragment images for a list of Morgan bit IDs.
    Bits are sorted by importance_vec descending.
    """
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    bits   = sorted(bit_list, key=lambda b: importance_vec[b], reverse=True)[:n]
    images, titles = [], []
    for bit in bits:
        img = get_bit_fragment_img(smiles_pool, bit)
        if img is None:
            img = PILImage.new('RGB', (200, 200), (240, 240, 240))
        images.append(img)
        titles.append(f'Bit {bit}\n|SHAP|={importance_vec[bit]:.4f}')

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.9))
    axes = np.array(axes).flatten()
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img); ax.set_title(t, fontsize=8); ax.axis('off')
    for ax in axes[len(images):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()



def three_way_scaffold_split(df, train_frac=0.70, val_frac=0.15, seed=SEED):
    """
    Split compounds into train/val/test by Murcko scaffold.
    Returns (tr_idx, va_idx, te_idx) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    scaffolds = df['scaffold'].values
    unique_sc = list(set(scaffolds))
    rng.shuffle(unique_sc)

    n      = len(df)
    n_val  = int(n * val_frac)
    n_test = int(n * (1 - train_frac - val_frac))

    val_sc, test_sc = set(), set()
    n_va, n_te = 0, 0
    for sc in unique_sc:
        cnt = (scaffolds == sc).sum()
        if n_te < n_test:
            test_sc.add(sc); n_te += cnt
        elif n_va < n_val:
            val_sc.add(sc);  n_va += cnt

    te_idx = np.where(np.isin(scaffolds, list(test_sc)))[0]
    va_idx = np.where(np.isin(scaffolds, list(val_sc)))[0]
    tr_idx = np.where(~np.isin(scaffolds, list(test_sc | val_sc)))[0]
    return tr_idx, va_idx, te_idx