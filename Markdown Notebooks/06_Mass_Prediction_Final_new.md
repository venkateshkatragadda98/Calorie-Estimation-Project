```python
# ===========================================
# CELL 1: Basic setup, imports, paths
# ===========================================
!pip install -q --upgrade scikit-learn

from google.colab import drive
drive.mount("/content/drive")

import os
import random
import time
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Project paths (UPDATED ROOT)
PROJECT_ROOT = Path("/content/drive/MyDrive/Cal_Estimation_Project")
N5K_ROOT     = PROJECT_ROOT / "data" / "raw" / "nutrition5k"

GEOMETRY_CSV    = PROJECT_ROOT / "cache" / "geometry_features.csv"
VISUAL_CSV      = PROJECT_ROOT / "cache" / "visual_features.csv"
DISHES_XLSX     = N5K_ROOT / "dishes.xlsx"
DISH_IMAGES_PKL = N5K_ROOT / "dish_images.pkl"

OUTPUT_DIR   = PROJECT_ROOT / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NEW: cache dir + helper
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def save_cache(obj, name: str):
    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved cache → {path}")

# Cache file for predictions (to avoid retraining)
PRED_PATH  = OUTPUT_DIR / "nutrition5k_dual_branch_predictions.csv"
MODEL_PATH = OUTPUT_DIR / "nutrition5k_dual_branch_model.pth"

print("Project root:", PROJECT_ROOT)
print("Output dir  :", OUTPUT_DIR)
print("Geometry CSV exists:", GEOMETRY_CSV.exists())
print("Visual   CSV exists:", VISUAL_CSV.exists())
print("Dishes  XLSX exists:", DISHES_XLSX.exists())
print("dish_images.pkl exists:", DISH_IMAGES_PKL.exists())
print("Cached prediction file exists:", PRED_PATH.exists())

```

    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.5/9.5 MB[0m [31m72.6 MB/s[0m eta [36m0:00:00[0m
    [?25hMounted at /content/drive
    Using device: cuda
    Project root: /content/drive/MyDrive/Cal_Estimation_Project
    Output dir  : /content/drive/MyDrive/Cal_Estimation_Project/Outputs
    Geometry CSV exists: True
    Visual   CSV exists: True
    Dishes  XLSX exists: True
    dish_images.pkl exists: True
    Cached prediction file exists: True



```python
# ===========================================
# CELL 2: Load metadata and feature tables
# ===========================================
# 1) Load dishes metadata (has mass, calories, macros)
dishes_df = pd.read_excel(DISHES_XLSX)
print("Raw dishes_df:", dishes_df.shape)
print("dishes_df columns:", list(dishes_df.columns))

# Normalize meta columns for targets
meta = dishes_df.copy()
meta = meta.rename(columns={
    "dish_id": "dish_id",
    "total_mass": "true_mass_g",
    "total_calories": "true_calories_kcal"
})
keep_meta_cols = ["dish_id"]
if "true_mass_g" in meta.columns:
    keep_meta_cols.append("true_mass_g")
if "true_calories_kcal" in meta.columns:
    keep_meta_cols.append("true_calories_kcal")

meta = meta[keep_meta_cols].copy()
meta["dish_id"] = meta["dish_id"].astype(str)
print("Normalized meta:", meta.shape)
print("meta columns:", list(meta.columns))

# 2) Load geometry features (MiDaS and segmentation based)
geometry_df = pd.read_csv(GEOMETRY_CSV)
geometry_df["dish_id"] = geometry_df["dish_id"].astype(str)
print("geometry_df:", geometry_df.shape)
print("geometry_df columns:", list(geometry_df.columns))

# 3) Load visual features (color stats + CLIP/ViT PCs)
visual_df = pd.read_csv(VISUAL_CSV)
visual_df["dish_id"] = visual_df["dish_id"].astype(str)
print("visual_df:", visual_df.shape)
print("visual_df columns:", list(visual_df.columns))

# 4) Load dish_images.pkl (RGB and depth bytes)
dish_images_df = pd.read_pickle(DISH_IMAGES_PKL)
print("dish_images_df:", dish_images_df.shape)
print("dish_images_df columns:", list(dish_images_df.columns))

# Make sure dish id is string
dish_images_df["dish"] = dish_images_df["dish"].astype(str)
dish_ids_with_img = set(dish_images_df["dish"].unique())
print("Unique dishes with images:", len(dish_ids_with_img))

```

    Raw dishes_df: (5006, 6)
    dishes_df columns: ['dish_id', 'total_mass', 'total_calories', 'total_fat', 'total_carb', 'total_protein']
    Normalized meta: (5006, 3)
    meta columns: ['dish_id', 'true_mass_g', 'true_calories_kcal']
    geometry_df: (3195, 13)
    geometry_df columns: ['dish_id', 'true_mass_g', 'true_calories_kcal', 'area_px', 'midas_volume', 'mean_height', 'median_height', 'max_height', 'std_height', 'avg_hue', 'avg_sat', 'avg_val', 'std_val']
    visual_df: (3195, 16)
    visual_df columns: ['dish_id', 'vis_pc1', 'vis_pc2', 'vis_pc3', 'vis_pc4', 'vis_pc5', 'vis_pc6', 'vis_pc7', 'vis_pc8', 'vis_pc9', 'vis_pc10', 'vis_pc11', 'vis_pc12', 'vis_pc13', 'vis_pc14', 'vis_pc15']
    dish_images_df: (3490, 3)
    dish_images_df columns: ['dish', 'rgb_image', 'depth_image']
    Unique dishes with images: 3490



```python
# ===========================================
# CELL 3: Merge feature tables and targets
# ===========================================
print("Merging geometry + visual on dish_id...")
df = geometry_df.merge(visual_df, on="dish_id", how="inner")
print("After geometry + visual merge:", df.shape)

# Merge meta if df does not already contain a usable mass column
mass_cols_in_df = [c for c in df.columns if "mass" in c.lower()]

if len(mass_cols_in_df) == 0:
    print("No mass column in geometry/visual; merging meta...")
    df = df.merge(meta, on="dish_id", how="inner")
else:
    print("Mass-like columns already present in geometry/visual:", mass_cols_in_df)
    df = df.merge(meta, on="dish_id", how="left")

print("After merging meta:", df.shape)

# Resolve possible duplicate mass / calories columns
mass_candidates = [c for c in df.columns if "true_mass" in c.lower() or "total_mass" in c.lower()]
cal_candidates  = [c for c in df.columns if "calories" in c.lower()]

print("Mass candidate columns:", mass_candidates)
print("Calorie candidate columns:", cal_candidates)

if "true_mass_g" in df.columns:
    df["target_mass_g"] = df["true_mass_g"]
elif len(mass_candidates) > 0:
    df["target_mass_g"] = df[mass_candidates].bfill(axis=1).iloc[:, 0]
else:
    raise AssertionError("No mass target column found in merged df.")

if "true_calories_kcal" in df.columns:
    df["target_cal_kcal"] = df["true_calories_kcal"]
elif len(cal_candidates) > 0:
    df["target_cal_kcal"] = df[cal_candidates].bfill(axis=1).iloc[:, 0]
else:
    df["target_cal_kcal"] = np.nan

# Keep only rows that have a corresponding RGB image in dish_images_df
df["has_image"] = df["dish_id"].isin(dish_ids_with_img)
df = df[df["has_image"]].copy()
df = df.drop(columns=["has_image"])
print("After restricting to dishes with RGB images:", df.shape)

# Drop rows with missing or non-positive mass
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target_mass_g"])
df = df[df["target_mass_g"] > 0].copy()
print("After dropping missing or non-positive mass:", df.shape)

print(
    "Mass stats: min={:.1f}g, max={:.1f}g, mean={:.1f}g, std={:.1f}g".format(
        df["target_mass_g"].min(),
        df["target_mass_g"].max(),
        df["target_mass_g"].mean(),
        df["target_mass_g"].std()
    )
)

df.head()

```

    Merging geometry + visual on dish_id...
    After geometry + visual merge: (3195, 28)
    Mass-like columns already present in geometry/visual: ['true_mass_g']
    After merging meta: (3195, 30)
    Mass candidate columns: ['true_mass_g_x', 'true_mass_g_y']
    Calorie candidate columns: ['true_calories_kcal_x', 'true_calories_kcal_y']
    After restricting to dishes with RGB images: (3195, 32)
    After dropping missing or non-positive mass: (3195, 32)
    Mass stats: min=1.1g, max=798.9g, mean=240.0g, std=186.0g






  <div id="df-1bcdff10-2f44-4908-8b88-813c10d75f8d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dish_id</th>
      <th>true_mass_g_x</th>
      <th>true_calories_kcal_x</th>
      <th>area_px</th>
      <th>midas_volume</th>
      <th>mean_height</th>
      <th>median_height</th>
      <th>max_height</th>
      <th>std_height</th>
      <th>avg_hue</th>
      <th>...</th>
      <th>vis_pc10</th>
      <th>vis_pc11</th>
      <th>vis_pc12</th>
      <th>vis_pc13</th>
      <th>vis_pc14</th>
      <th>vis_pc15</th>
      <th>true_mass_g_y</th>
      <th>true_calories_kcal_y</th>
      <th>target_mass_g</th>
      <th>target_cal_kcal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dish_1556572657</td>
      <td>41.399998</td>
      <td>36.0</td>
      <td>227737</td>
      <td>7339248.0</td>
      <td>32.226860</td>
      <td>27.706909</td>
      <td>187.235687</td>
      <td>38.267372</td>
      <td>50.180318</td>
      <td>...</td>
      <td>0.625997</td>
      <td>-0.825387</td>
      <td>0.082035</td>
      <td>-0.084148</td>
      <td>-1.042391</td>
      <td>0.515940</td>
      <td>41.399998</td>
      <td>36</td>
      <td>41.399998</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dish_1556573514</td>
      <td>6.440000</td>
      <td>23.0</td>
      <td>231435</td>
      <td>10960611.0</td>
      <td>47.359348</td>
      <td>30.816589</td>
      <td>309.500061</td>
      <td>64.410133</td>
      <td>50.208439</td>
      <td>...</td>
      <td>0.382944</td>
      <td>-0.768417</td>
      <td>0.130779</td>
      <td>-0.415776</td>
      <td>0.381437</td>
      <td>0.314428</td>
      <td>6.440000</td>
      <td>23</td>
      <td>6.440000</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dish_1556575014</td>
      <td>71.299995</td>
      <td>62.0</td>
      <td>230526</td>
      <td>5786524.0</td>
      <td>25.101395</td>
      <td>1.286407</td>
      <td>192.026001</td>
      <td>42.333065</td>
      <td>52.111371</td>
      <td>...</td>
      <td>-1.008527</td>
      <td>0.035902</td>
      <td>-0.036344</td>
      <td>-0.694468</td>
      <td>-1.043567</td>
      <td>-0.380548</td>
      <td>71.299995</td>
      <td>62</td>
      <td>71.299995</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dish_1556575083</td>
      <td>27.520000</td>
      <td>64.0</td>
      <td>232512</td>
      <td>12251497.0</td>
      <td>52.691891</td>
      <td>34.290070</td>
      <td>199.190979</td>
      <td>57.741058</td>
      <td>54.059859</td>
      <td>...</td>
      <td>0.716415</td>
      <td>-1.075433</td>
      <td>0.756044</td>
      <td>-1.012301</td>
      <td>0.407430</td>
      <td>0.420339</td>
      <td>27.520000</td>
      <td>64</td>
      <td>27.520000</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dish_1556575124</td>
      <td>4.480000</td>
      <td>28.0</td>
      <td>239885</td>
      <td>13736331.0</td>
      <td>57.262150</td>
      <td>19.864655</td>
      <td>259.285706</td>
      <td>67.117287</td>
      <td>57.369181</td>
      <td>...</td>
      <td>1.475376</td>
      <td>1.078840</td>
      <td>0.077915</td>
      <td>1.656281</td>
      <td>-1.249205</td>
      <td>-0.405906</td>
      <td>4.480000</td>
      <td>28</td>
      <td>4.480000</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1bcdff10-2f44-4908-8b88-813c10d75f8d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1bcdff10-2f44-4908-8b88-813c10d75f8d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1bcdff10-2f44-4908-8b88-813c10d75f8d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-20437d4d-ae5c-4642-a182-ed4256f00f00">
      <button class="colab-df-quickchart" onclick="quickchart('df-20437d4d-ae5c-4642-a182-ed4256f00f00')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-20437d4d-ae5c-4642-a182-ed4256f00f00 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# ===========================================
# CELL 4: Define feature lists and helper functions
# ===========================================
exclude_cols = {
    "dish_id",
    "target_mass_g",
    "target_cal_kcal"
}

tabular_cols = [c for c in df.columns
                if c not in exclude_cols and df[c].dtype != "O"]

print("Number of tabular feature columns:", len(tabular_cols))

core_geom_expected = ["area_px", "midas_volume", "mean_height"]
for col in core_geom_expected:
    if col not in tabular_cols and col in df.columns:
        tabular_cols.append(col)

tabular_cols = sorted(set(tabular_cols))
print("Final tabular feature count:", len(tabular_cols))

def compute_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def compute_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

```

    Number of tabular feature columns: 29
    Final tabular feature count: 29



```python
# ===========================================
# CELL 5: Dataset and transforms
# ===========================================
IMG_SIZE = 256

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_rgb_from_pkl(dish_id_str):
    row = dish_images_df[dish_images_df["dish"] == dish_id_str]
    if len(row) == 0:
        return None
    rgb_bytes = row["rgb_image"].iloc[0]
    img = Image.open(BytesIO(rgb_bytes)).convert("RGB")
    return img

class Nutrition5kDataset(Dataset):
    def __init__(self, df, tabular_cols, transform=None, scaler=None):
        self.df = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.transform = transform
        self.scaler = scaler

        self.X_tab = self.df[self.tabular_cols].values.astype(np.float32)
        if self.scaler is not None:
            self.X_tab = self.scaler.transform(self.X_tab)

        self.y = self.df["target_mass_g"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id_str = str(row["dish_id"])

        img = load_rgb_from_pkl(dish_id_str)
        if img is None:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        tab_feats = self.X_tab[idx]
        tab_feats_tensor = torch.from_numpy(tab_feats)

        target = self.y[idx]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return img_tensor, tab_feats_tensor, target_tensor

```


```python
# ===========================================
# CELL 6: Dual-branch model (CNN + tabular)
# ===========================================
class VisionBackbone(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, out_dim=256):
        super().__init__()
        if backbone_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            modules = list(base.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            in_feats = base.fc.in_features
        elif backbone_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            modules = list(base.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            in_feats = base.fc.in_features
        else:
            raise ValueError("Unsupported backbone")

        self.fc = nn.Sequential(
            nn.Linear(in_feats, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class TabularBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class FusionRegressor(nn.Module):
    def __init__(self, num_tabular_features, vision_embed_dim=256, tab_embed_dim=128):
        super().__init__()
        self.vision = VisionBackbone(backbone_name="resnet50", pretrained=True, out_dim=vision_embed_dim)
        self.tabular = TabularBranch(in_dim=num_tabular_features, hidden_dim=256, out_dim=tab_embed_dim)

        fusion_in = vision_embed_dim + tab_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img, tab):
        v = self.vision(img)
        t = self.tabular(tab)
        x = torch.cat([v, t], dim=1)
        out = self.head(x)
        return out.squeeze(1)

```


```python
# ===========================================
# CELL 7: Train / eval utilities
# ===========================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n = 0

    for imgs, tabs, targets in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs, tabs)
        loss = F.l1_loss(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_mae += torch.abs(preds - targets).sum().item()
        n += batch_size

    return running_loss / n, running_mae / n

@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    preds_all = []
    targets_all = []
    running_mae = 0.0
    n = 0

    for imgs, tabs, targets in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(imgs, tabs)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

        batch_size = targets.size(0)
        running_mae += torch.abs(preds - targets).sum().item()
        n += batch_size

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    mae = running_mae / n
    rmse = compute_rmse(targets_all, preds_all)
    mape = compute_mape(targets_all, preds_all)

    return mae, rmse, mape, preds_all, targets_all

```


```python
# ===========================================
# CELL 8: Train/val/test split and scaler fit
# ===========================================
dish_ids = df["dish_id"].values
targets = df["target_mass_g"].values

mass_bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.15,
    random_state=SEED,
    stratify=mass_bins
)

df_trainval = df.iloc[train_idx].reset_index(drop=True)
df_test     = df.iloc[test_idx].reset_index(drop=True)

print("Train+Val size:", len(df_trainval))
print("Test size     :", len(df_test))

scaler = StandardScaler()
scaler.fit(df_trainval[tabular_cols].values.astype(np.float32))

```

    Train+Val size: 2715
    Test size     : 480





<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>StandardScaler()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_mean&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_std&nbsp;</td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




```python
# ===========================================
# CELL 10: Train final model with caching of BEST test MAE
# ===========================================
BATCH_SIZE = 32
LR = 1e-4
EPOCHS_FINAL = 20

HAS_MODEL = MODEL_PATH.exists()
HAS_PREDS = PRED_PATH.exists()

if HAS_MODEL and HAS_PREDS:
    print("Found cached model and predictions. Loading best test MAE from disk...")

    # Load cached predictions
    results = pd.read_csv(PRED_PATH)
    results["abs_error_g"] = np.abs(results["true_mass_g"] - results["pred_mass_g"])

    cached_best_mae = results["abs_error_g"].mean()
    test_mae = cached_best_mae  # <-- FIX so test_mae is always defined

    print("Loaded cached BEST test MAE: {:.2f} g".format(cached_best_mae))

    # Load model safely
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model_final = FusionRegressor(num_tabular_features=len(tabular_cols)).to(DEVICE)
    model_final.load_state_dict(checkpoint["state_dict"])

else:
    print("No cached model found. Training NOW and saving best test MAE...")

    train_ds_full = Nutrition5kDataset(df_trainval, tabular_cols, transform=train_transform, scaler=scaler)
    test_ds       = Nutrition5kDataset(df_test, tabular_cols, transform=val_transform, scaler=scaler)

    train_loader_full = DataLoader(
        train_ds_full, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    model_final = FusionRegressor(num_tabular_features=len(tabular_cols)).to(DEVICE)

    # Freeze early CNN layers
    for name, param in model_final.vision.cnn.named_parameters():
        if "layer3" not in name and "layer4" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_final.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_test_mae = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS_FINAL + 1):
        start_time = time.time()
        train_loss, train_mae = train_one_epoch(model_final, train_loader_full, optimizer)
        test_mae, test_rmse, test_mape, test_preds, test_targets = eval_one_epoch(model_final, test_loader)
        scheduler.step(test_mae)

        elapsed = time.time() - start_time
        print(
            "Final model | Epoch {:02d}/{:02d} | train_loss {:.3f} | train_MAE {:.2f} | "
            "test_MAE {:.2f} | test_RMSE {:.2f} | test_MAPE {:.1f}% | time {:.1f}s".format(
                epoch, EPOCHS_FINAL, train_loss, train_mae, test_mae, test_rmse, test_mape, elapsed
            )
        )

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_state = model_final.state_dict()

    print("\nBest test MAE achieved during training: {:.2f} g".format(best_test_mae))

    # Recompute predictions using the best state
    model_final.load_state_dict(best_state)
    test_mae, test_rmse, test_mape, test_preds, test_targets = eval_one_epoch(model_final, test_loader)

    results = pd.DataFrame({
        "dish_id": df_test["dish_id"].values,
        "true_mass_g": test_targets,
        "pred_mass_g": test_preds
    })
    results["abs_error_g"] = np.abs(results["true_mass_g"] - results["pred_mass_g"])

    # Save best predictions
    results.to_csv(PRED_PATH, index=False)
    print("Saved best predictions to:", PRED_PATH)

    # Save best model
    torch.save(
        {
            "state_dict": best_state,
            "tabular_cols": tabular_cols
        },
        MODEL_PATH
    )
    print("Saved best model to:", MODEL_PATH)

```

    Found cached model and predictions. Loading best test MAE from disk...
    Loaded cached BEST test MAE: 23.40 g



```python
# ===========================================
# CELL 11: Simple plots for analysis (optional)
# ===========================================
print("results shape:", results.shape)
print("results columns:", list(results.columns))

plt.figure(figsize=(6, 6))
plt.scatter(results["true_mass_g"], results["pred_mass_g"], alpha=0.5)
min_val = min(results["true_mass_g"].min(), results["pred_mass_g"].min())
max_val = max(results["true_mass_g"].max(), results["pred_mass_g"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
plt.xlabel("True mass (g)")
plt.ylabel("Predicted mass (g)")
plt.title("True vs Predicted Mass (Test)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(results["abs_error_g"], bins=40, edgecolor="black")
plt.xlabel("Absolute error (g)")
plt.ylabel("Count")
plt.title("Error distribution (Test)")
plt.tight_layout()
plt.show()

```

    results shape: (480, 4)
    results columns: ['dish_id', 'true_mass_g', 'pred_mass_g', 'abs_error_g']



    
![png](06_Mass_Prediction_Final_new_files/06_Mass_Prediction_Final_new_9_1.png)
    



    
![png](06_Mass_Prediction_Final_new_files/06_Mass_Prediction_Final_new_9_2.png)
    



```python
# ===========================================
# CELL 12: Visualize test images with true and predicted mass
# ===========================================
vis_df = results.copy()

N = 12
N = min(N, len(vis_df))

sample_df = vis_df.sample(N, random_state=SEED).reset_index(drop=True)

cols = 4
rows = int(np.ceil(N / cols))

plt.figure(figsize=(4 * cols, 4 * rows))

for idx in range(N):
    dish_id = str(sample_df.loc[idx, "dish_id"])
    true_m  = float(sample_df.loc[idx, "true_mass_g"])
    pred_m  = float(sample_df.loc[idx, "pred_mass_g"])

    img = load_rgb_from_pkl(dish_id)
    if img is None:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))

    ax = plt.subplot(rows, cols, idx + 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Dish: {}\nTrue: {:.1f} g | Pred: {:.1f} g".format(dish_id, true_m, pred_m),
        fontsize=10
    )

plt.tight_layout()
plt.show()

```


    
![png](06_Mass_Prediction_Final_new_files/06_Mass_Prediction_Final_new_10_0.png)
    



```python
# ===========================================
# CELL 13: Save artifacts for follow-up notebook
# ===========================================
ARTIFACT_DIR = OUTPUT_DIR / "nutrition5k_mass_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving artifacts to:", ARTIFACT_DIR)

# 1) Save trainval and test feature tables
trainval_path = ARTIFACT_DIR / "trainval_features.csv"
testfeat_path = ARTIFACT_DIR / "test_features.csv"

df_trainval.to_csv(trainval_path, index=False)
df_test.to_csv(testfeat_path, index=False)
print("Saved trainval features to:", trainval_path)
print("Saved test features to    :", testfeat_path)

# 2) Save indices of the split
train_idx_path = ARTIFACT_DIR / "train_indices.npy"
test_idx_path  = ARTIFACT_DIR / "test_indices.npy"

np.save(train_idx_path, train_idx)
np.save(test_idx_path, test_idx)
print("Saved train indices to:", train_idx_path)
print("Saved test indices to :", test_idx_path)

# 3) Save the tabular scaler
scaler_path = ARTIFACT_DIR / "tabular_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print("Saved StandardScaler to:", scaler_path)

# 4) Save tabular feature names
tab_cols_path = ARTIFACT_DIR / "tabular_feature_names.txt"
with open(tab_cols_path, "w") as f:
    for col in tabular_cols:
        f.write(col + "\n")
print("Saved tabular feature names to:", tab_cols_path)

# 5) Build a test results table with mass predictions and meta information
macro_keywords = ["protein", "prot", "carb", "fat"]
macro_cols = [
    c for c in dishes_df.columns
    if any(k in c.lower() for k in macro_keywords)
]

macro_meta_cols = ["dish_id"] + macro_cols if macro_cols else ["dish_id"]
macro_meta = dishes_df[macro_meta_cols].copy()
macro_meta["dish_id"] = macro_meta["dish_id"].astype(str)

meta_for_join = meta.copy()
meta_for_join["dish_id"] = meta_for_join["dish_id"].astype(str)

results_for_join = results.copy()
results_for_join["dish_id"] = results_for_join["dish_id"].astype(str)

test_summary = (
    df_test[["dish_id"]].copy()
    .merge(results_for_join, on="dish_id", how="left")
    .merge(meta_for_join, on="dish_id", how="left", suffixes=("", "_meta"))
    .merge(macro_meta, on="dish_id", how="left")
)

test_summary_path = ARTIFACT_DIR / "test_mass_predictions_with_meta.csv"
test_summary.to_csv(test_summary_path, index=False)
print("Saved test mass predictions with meta to:", test_summary_path)

print("\nSummary for follow-up notebook:")
print("Rows in test_summary:", len(test_summary))
print("Columns in test_summary:", list(test_summary.columns))
print("Current best test MAE (this run): {:.2f} g".format(test_mae))
print("Model weights are saved at:", MODEL_PATH)
print("Raw prediction CSV is at  :", PRED_PATH)

```

    Saving artifacts to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts
    Saved trainval features to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/trainval_features.csv
    Saved test features to    : /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/test_features.csv
    Saved train indices to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/train_indices.npy
    Saved test indices to : /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/test_indices.npy
    Saved StandardScaler to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/tabular_scaler.pkl
    Saved tabular feature names to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/tabular_feature_names.txt
    Saved test mass predictions with meta to: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_mass_artifacts/test_mass_predictions_with_meta.csv
    
    Summary for follow-up notebook:
    Rows in test_summary: 480
    Columns in test_summary: ['dish_id', 'true_mass_g', 'pred_mass_g', 'abs_error_g', 'true_mass_g_meta', 'true_calories_kcal', 'total_fat', 'total_carb', 'total_protein']
    Current best test MAE (this run): 23.40 g
    Model weights are saved at: /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_dual_branch_model.pth
    Raw prediction CSV is at  : /content/drive/MyDrive/Cal_Estimation_Project/Outputs/nutrition5k_dual_branch_predictions.csv



```python
# ===========================================
# CELL 14: Save lightweight cache for fast re-runs
# ===========================================
cache_data = {
    "tabular_cols": tabular_cols,
    "train_size": len(df_trainval),
    "test_size": len(df_test),
    "results_shape": results.shape if "results" in globals() else None,
    "pred_csv": str(PRED_PATH),
    "model_path": str(MODEL_PATH),
    "artifact_dir": str(ARTIFACT_DIR),
    "best_test_mae": float(test_mae) if "test_mae" in globals() else None,
}

save_cache(cache_data, "06_part1_final")

```

    ✔ Saved cache → /content/drive/MyDrive/Cal_Estimation_Project/cache/06_part1_final.pkl

