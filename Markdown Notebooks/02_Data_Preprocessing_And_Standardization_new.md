```python
# Setup and import all required libraries for data preprocessing.
# This includes standard libraries for image processing, data handling, and file management.

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define base project paths
BASE_DIR = Path("/content/drive/MyDrive/Cal_Estimation_Project/data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Drive mounted and project directories ready.")
print("Raw data path:", RAW_DIR)
print("Processed data path:", PROCESSED_DIR)

```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    Drive mounted and project directories ready.
    Raw data path: /content/drive/MyDrive/Cal_Estimation_Project/data/raw
    Processed data path: /content/drive/MyDrive/Cal_Estimation_Project/data/processed



```python
# ============================================================
# CACHE UTILITIES FOR NOTEBOOK 02
# Saves processed intermediate datasets for faster re-runs.
# ============================================================
import pickle   # <-- REQUIRED
CACHE_DIR = "/content/drive/MyDrive/Cal_Estimation_Project/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(obj, name="02_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved cache → {path}")

def load_cache(name="02_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    if os.path.exists(path):
        print(f"✔ Loaded cache → {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

```


```python
# Load the dataset manifest created in Notebook 01.
from pathlib import Path
import json

possible_paths = [
    Path("/content/drive/MyDrive/Cal_Estimation_Project/manifests/datasets_manifest.json"),
    Path("/content/drive/MyDrive/Cal_Estimation_Project/data/manifests/datasets_manifest.json")
]

manifest_path = next((p for p in possible_paths if p.exists()), None)

if manifest_path is None:
    raise FileNotFoundError("Manifest not found. Please verify that Notebook 01 was run and manifests folder exists.")
else:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    print(f"Loaded manifest from: {manifest_path}\n")
    print(json.dumps(manifest, indent=4))

# Extract dataset paths
foodseg_path = Path(manifest["FoodSeg103"])
nutrition5k_path = Path(manifest["Nutrition5k"])
usda_path = Path(manifest["USDA_FoodData"])

```

    Loaded manifest from: /content/drive/MyDrive/Cal_Estimation_Project/data/manifests/datasets_manifest.json
    
    {
        "FoodSeg103": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103",
        "Nutrition5k": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k",
        "USDA_FoodData": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda"
    }



```python
def verify_dataset(name, path):
    if path.exists():
        files = list(path.glob("**/*"))
        print(f"{name}: {len(files)} files found at {path}")
    else:
        print(f"{name}: Folder not found at {path}")

print("Verifying dataset paths...\n")
verify_dataset("FoodSeg103", foodseg_path)
verify_dataset("Nutrition5k", nutrition5k_path)
verify_dataset("USDA FoodData", usda_path)
print("\nAll datasets verified.")

```

    Verifying dataset paths...
    
    FoodSeg103: 14250 files found at /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103
    Nutrition5k: 6 files found at /content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k
    USDA FoodData: 25 files found at /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda
    
    All datasets verified.



```python
from PIL import Image
import random

img_dir = foodseg_path / "Images" / "img_dir" / "train"
mask_dir = foodseg_path / "Images" / "ann_dir" / "train"

image_files = sorted(list(img_dir.glob("*.jpg")))
mask_files = sorted(list(mask_dir.glob("*.png")))

print(f"Found {len(image_files)} training images and {len(mask_files)} masks.")

subset_pairs = [(img, mask_dir / (img.stem + ".png")) for img in image_files[:50]]

def load_image_pair(img_path, mask_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    return img, mask

sample_img, sample_mask = load_image_pair(*subset_pairs[0])

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(sample_img); plt.title("RGB Sample"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(sample_mask, cmap="nipy_spectral"); plt.title("Segmentation Mask"); plt.axis("off")
plt.tight_layout(); plt.show()

```

    Found 4983 training images and 4983 masks.



    
![png](02_Data_Preprocessing_And_Standardization_new_files/02_Data_Preprocessing_And_Standardization_new_4_1.png)
    



```python
from PIL import Image
import io

nutrition_df = pd.read_excel(nutrition5k_path / "dishes.xlsx")
dish_pkl = pd.read_pickle(nutrition5k_path / "dish_images.pkl")

def decode_image_bytes(image_bytes):
    return np.array(Image.open(io.BytesIO(image_bytes)))

decoded_samples = []
for i in range(5):
    rgb_img = decode_image_bytes(dish_pkl.iloc[i]["rgb_image"])
    depth_img = decode_image_bytes(dish_pkl.iloc[i]["depth_image"])
    decoded_samples.append((rgb_img, depth_img))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(decoded_samples[0][0]); plt.title("Nutrition5k RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(decoded_samples[0][1], cmap="gray"); plt.title("Nutrition5k Depth"); plt.axis("off")
plt.tight_layout(); plt.show()

```


    
![png](02_Data_Preprocessing_And_Standardization_new_files/02_Data_Preprocessing_And_Standardization_new_5_0.png)
    



```python
food_csv = usda_path / "food.csv"
nutrient_csv = usda_path / "nutrient.csv"
food_nutrient_csv = usda_path / "food_nutrient.csv"

df_food = pd.read_csv(food_csv, usecols=["fdc_id", "description", "food_category_id"])
df_nutrient = pd.read_csv(nutrient_csv, usecols=["id", "name", "unit_name"])
df_food_nutrient = pd.read_csv(food_nutrient_csv, usecols=["fdc_id", "nutrient_id", "amount"])

merged = (df_food_nutrient
          .merge(df_nutrient, left_on="nutrient_id", right_on="id")
          .merge(df_food, on="fdc_id"))

key_nutrients = ["Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference"]
df_usda_filtered = merged[merged["name"].isin(key_nutrients)]

print(f"Filtered USDA nutrient table: {df_usda_filtered.shape}")
display(df_usda_filtered.head())

```

    Filtered USDA nutrient table: (6557, 8)




  <div id="df-8e105fed-6400-4ae2-a6c7-e891d6d1ec9f" class="colab-df-container">
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
      <th>fdc_id</th>
      <th>nutrient_id</th>
      <th>amount</th>
      <th>id</th>
      <th>name</th>
      <th>unit_name</th>
      <th>description</th>
      <th>food_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>319877</td>
      <td>1004</td>
      <td>19.0</td>
      <td>1004</td>
      <td>Total lipid (fat)</td>
      <td>G</td>
      <td>Hummus</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>319882</td>
      <td>1004</td>
      <td>18.7</td>
      <td>1004</td>
      <td>Total lipid (fat)</td>
      <td>G</td>
      <td>Hummus</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>319892</td>
      <td>1004</td>
      <td>16.6</td>
      <td>1004</td>
      <td>Total lipid (fat)</td>
      <td>G</td>
      <td>Hummus</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>319899</td>
      <td>1004</td>
      <td>19.1</td>
      <td>1004</td>
      <td>Total lipid (fat)</td>
      <td>G</td>
      <td>Hummus</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>319908</td>
      <td>1004</td>
      <td>18.2</td>
      <td>1004</td>
      <td>Total lipid (fat)</td>
      <td>G</td>
      <td>Hummus</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8e105fed-6400-4ae2-a6c7-e891d6d1ec9f')"
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
        document.querySelector('#df-8e105fed-6400-4ae2-a6c7-e891d6d1ec9f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8e105fed-6400-4ae2-a6c7-e891d6d1ec9f');
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


    <div id="df-ebd333a8-bd2d-438e-86ad-4b211560febd">
      <button class="colab-df-quickchart" onclick="quickchart('df-ebd333a8-bd2d-438e-86ad-4b211560febd')"
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
            document.querySelector('#df-ebd333a8-bd2d-438e-86ad-4b211560febd button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
df_usda_std = df_usda_filtered.loc[:, ["fdc_id", "description", "name", "amount", "unit_name"]].copy()

def standardize_units(row):
    if row["name"] == "Energy" and str(row["unit_name"]).lower() == "kj":
        return row["amount"] * 0.239006
    return row["amount"]

df_usda_std["amount_std"] = df_usda_std.apply(standardize_units, axis=1)

df_usda_pivot = (
    df_usda_std
    .pivot_table(index=["fdc_id", "description"],
                 columns="name", values="amount_std", aggfunc="mean")
    .reset_index()
)

df_usda_pivot.rename(columns={
    "Energy": "calories_kcal",
    "Protein": "protein_g",
    "Total lipid (fat)": "fat_g",
    "Carbohydrate, by difference": "carbs_g"
}, inplace=True)

print("Normalized USDA reference data (first 5 rows):")
display(df_usda_pivot.head())

```

    Normalized USDA reference data (first 5 rows):




  <div id="df-963670cc-22ee-4d57-99e9-8e5a2e767247" class="colab-df-container">
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
      <th>name</th>
      <th>fdc_id</th>
      <th>description</th>
      <th>carbs_g</th>
      <th>calories_kcal</th>
      <th>protein_g</th>
      <th>fat_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>319877</td>
      <td>Hummus</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>319882</td>
      <td>Hummus</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>319892</td>
      <td>Hummus</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319899</td>
      <td>Hummus</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>319908</td>
      <td>Hummus</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-963670cc-22ee-4d57-99e9-8e5a2e767247')"
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
        document.querySelector('#df-963670cc-22ee-4d57-99e9-8e5a2e767247 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-963670cc-22ee-4d57-99e9-8e5a2e767247');
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


    <div id="df-10c21e13-d942-489a-b9fa-55eeb80f3f78">
      <button class="colab-df-quickchart" onclick="quickchart('df-10c21e13-d942-489a-b9fa-55eeb80f3f78')"
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
            document.querySelector('#df-10c21e13-d942-489a-b9fa-55eeb80f3f78 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
complete_usda = df_usda_pivot.dropna(subset=["calories_kcal", "protein_g", "fat_g", "carbs_g"])
print(f"{len(complete_usda)} foods have complete macronutrient information.")
display(complete_usda.sample(5, random_state=42))

```

    135 foods have complete macronutrient information.




  <div id="df-6cbb06b8-d693-45e6-ab0c-e0915fcae108" class="colab-df-container">
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
      <th>name</th>
      <th>fdc_id</th>
      <th>description</th>
      <th>carbs_g</th>
      <th>calories_kcal</th>
      <th>protein_g</th>
      <th>fat_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2152</th>
      <td>746783</td>
      <td>Sausage, turkey, breakfast links, mild, raw</td>
      <td>0.93</td>
      <td>168.869118</td>
      <td>16.70</td>
      <td>10.40</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>334849</td>
      <td>Beef, loin, top loin steak, boneless, lip-on, ...</td>
      <td>0.00</td>
      <td>154.937944</td>
      <td>22.80</td>
      <td>6.39</td>
    </tr>
    <tr>
      <th>2318</th>
      <td>747693</td>
      <td>Ketchup, restaurant</td>
      <td>26.80</td>
      <td>116.817464</td>
      <td>1.11</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>370</th>
      <td>324860</td>
      <td>Peanut butter, smooth style, with salt</td>
      <td>22.30</td>
      <td>597.257500</td>
      <td>22.50</td>
      <td>51.10</td>
    </tr>
    <tr>
      <th>725</th>
      <td>329490</td>
      <td>Egg, whole, dried</td>
      <td>1.87</td>
      <td>575.502230</td>
      <td>48.10</td>
      <td>39.80</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6cbb06b8-d693-45e6-ab0c-e0915fcae108')"
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
        document.querySelector('#df-6cbb06b8-d693-45e6-ab0c-e0915fcae108 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6cbb06b8-d693-45e6-ab0c-e0915fcae108');
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


    <div id="df-61fde921-b080-4530-94a9-d88e54898b27">
      <button class="colab-df-quickchart" onclick="quickchart('df-61fde921-b080-4530-94a9-d88e54898b27')"
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
            document.querySelector('#df-61fde921-b080-4530-94a9-d88e54898b27 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
import cv2
import numpy as np
from tqdm import tqdm

processed_foodseg = PROCESSED_DIR / "foodseg103_samples.npz"
processed_nutrition5k = PROCESSED_DIR / "nutrition5k_samples.npz"
processed_usda = PROCESSED_DIR / "usda_reference.csv"

TARGET_SIZE = (256, 256)

def resize_pair(image, mask, size=TARGET_SIZE):
    img_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized

resized_imgs, resized_masks = [], []
for img_path, mask_path in tqdm(subset_pairs, desc="Resizing FoodSeg103 samples"):
    img, mask = load_image_pair(img_path, mask_path)
    img_r, mask_r = resize_pair(img, mask)
    resized_imgs.append(img_r)
    resized_masks.append(mask_r)

np.savez_compressed(processed_foodseg,
                    images=np.array(resized_imgs),
                    masks=np.array(resized_masks))

np.savez_compressed(
    processed_nutrition5k,
    rgb=[x[0] for x in decoded_samples],
    depth=[x[1] for x in decoded_samples]
)

complete_usda.to_csv(processed_usda, index=False)

print("✅ Processed datasets saved successfully:")
print(f"- FoodSeg103 → {processed_foodseg}")
print(f"- Nutrition5k → {processed_nutrition5k}")
print(f"- USDA reference → {processed_usda}")

```

    Resizing FoodSeg103 samples: 100%|██████████| 50/50 [00:01<00:00, 29.85it/s]


    ✅ Processed datasets saved successfully:
    - FoodSeg103 → /content/drive/MyDrive/Cal_Estimation_Project/data/processed/foodseg103_samples.npz
    - Nutrition5k → /content/drive/MyDrive/Cal_Estimation_Project/data/processed/nutrition5k_samples.npz
    - USDA reference → /content/drive/MyDrive/Cal_Estimation_Project/data/processed/usda_reference.csv



```python
if processed_foodseg.exists():
    data = np.load(processed_foodseg)
    print("FoodSeg103:", data["images"].shape, "images,", data["masks"].shape, "masks")

if processed_nutrition5k.exists():
    data = np.load(processed_nutrition5k)
    print("Nutrition5k:", len(data["rgb"]), "RGB images,", len(data["depth"]), "depth maps")

if processed_usda.exists():
    df = pd.read_csv(processed_usda)
    print("USDA Reference:", df.shape, "records")
    display(df.head())

```

    FoodSeg103: (50, 256, 256, 3) images, (50, 256, 256) masks
    Nutrition5k: 5 RGB images, 5 depth maps
    USDA Reference: (135, 6) records




  <div id="df-27553cf2-fa83-4261-946d-5fcb6e68031f" class="colab-df-container">
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
      <th>fdc_id</th>
      <th>description</th>
      <th>carbs_g</th>
      <th>calories_kcal</th>
      <th>protein_g</th>
      <th>fat_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>321358</td>
      <td>Hummus, commercial</td>
      <td>14.90</td>
      <td>229.222880</td>
      <td>7.35</td>
      <td>17.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>321359</td>
      <td>Milk, reduced fat, fluid, 2% milkfat, with add...</td>
      <td>4.91</td>
      <td>49.976127</td>
      <td>3.35</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>321360</td>
      <td>Tomatoes, grape, raw</td>
      <td>5.51</td>
      <td>27.003839</td>
      <td>0.83</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>321611</td>
      <td>Beans, snap, green, canned, regular pack, drai...</td>
      <td>4.11</td>
      <td>20.777258</td>
      <td>1.04</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>321900</td>
      <td>Broccoli, raw</td>
      <td>6.29</td>
      <td>31.774396</td>
      <td>2.57</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-27553cf2-fa83-4261-946d-5fcb6e68031f')"
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
        document.querySelector('#df-27553cf2-fa83-4261-946d-5fcb6e68031f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-27553cf2-fa83-4261-946d-5fcb6e68031f');
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


    <div id="df-9f098256-bf5e-4f3a-ac41-23fcf2ee831a">
      <button class="colab-df-quickchart" onclick="quickchart('df-9f098256-bf5e-4f3a-ac41-23fcf2ee831a')"
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
            document.querySelector('#df-9f098256-bf5e-4f3a-ac41-23fcf2ee831a button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# ============================================================
# SAVE CACHE FOR NOTEBOOK 02 (NEW)
# ============================================================

cache_data = {
    "foodseg_samples_path": str(processed_foodseg),
    "nutrition5k_samples_path": str(processed_nutrition5k),
    "usda_reference_path": str(processed_usda),
    "num_foodseg_samples": len(resized_imgs),
    "num_nutrition_samples": len(decoded_samples)
}

save_cache(cache_data, "02_cache")

```

    ✔ Saved cache → /content/drive/MyDrive/Cal_Estimation_Project/cache/02_cache.pkl

