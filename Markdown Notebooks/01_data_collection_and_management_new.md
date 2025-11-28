```python
# ================================================
# Create Cal_Estimation_Project Folder Structure
# ================================================

import os

ROOT = "/content/drive/MyDrive/Cal_Estimation_Project"

folders = [
    f"{ROOT}",
    f"{ROOT}/data",
    f"{ROOT}/data/raw",
    f"{ROOT}/data/raw/foodseg103",
    f"{ROOT}/data/raw/nutrition5k",
    f"{ROOT}/data/raw/usda",
    f"{ROOT}/data/processed",
    f"{ROOT}/checkpoints",
    f"{ROOT}/outputs",
    f"{ROOT}/cache",
    f"{ROOT}/logs",
    f"{ROOT}/notebooks"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✔️ Project structure created successfully at:")
print(ROOT)

```

    ✔️ Project structure created successfully at:
    /content/drive/MyDrive/Cal_Estimation_Project



```python
# Mount Google Drive, upload Kaggle API key, and install Kaggle CLI for dataset access
from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

!pip install -q kaggle
!kaggle --version
%cd /content

```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).




     <input type="file" id="files-fc48304f-c939-4c17-b703-9560281818b8" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-fc48304f-c939-4c17-b703-9560281818b8">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving kaggle.json to kaggle.json
    Kaggle API 1.7.4.5
    /content



```python
# ============================================================
# CACHE UTILITIES (NEW CELL ADDED FOR NEW PROJECT STRUCTURE)
# Saves lightweight information needed for future notebooks.
# ============================================================

import pickle
import os

CACHE_DIR = "/content/drive/MyDrive/Cal_Estimation_Project/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(obj, name="01_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved cache → {path}")

def load_cache(name="01_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    if os.path.exists(path):
        print(f"✔ Loaded cache → {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

```


```python
# Define Kaggle dataset identifiers and create base folder to store raw datasets
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/Cal_Estimation_Project/data/raw")
kaggle_datasets = {
    "foodseg103": "ggrill/foodseg103",
    "nutrition5k": "siddhantrout/nutrition5k-dataset"
}

BASE_DIR.mkdir(parents=True, exist_ok=True)
print("Data root directory:", BASE_DIR)

```

    Data root directory: /content/drive/MyDrive/Cal_Estimation_Project/data/raw



```python
# Move uploaded Kaggle key to the correct directory and set permissions for authentication
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("Kaggle API key configured successfully.")

```

    Kaggle API key configured successfully.



```python
import os
import zipfile
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/Cal_Estimation_Project/data/raw")
BASE_DIR.mkdir(parents=True, exist_ok=True)

kaggle_datasets = {
    "foodseg103": "ggrill/foodseg103",
    "nutrition5k": "siddhantrout/nutrition5k-dataset",
    "nutrition5k_side": "zygmuntyt/nutrition5k-dataset-side-angle-images"
}

def expected_ok(name: str, dest: Path) -> bool:
    if name == "foodseg103":
        return (dest / "FoodSeg103" / "Images" / "img_dir" / "train").exists()

    if name == "nutrition5k":
        return (dest / "dish_images.pkl").exists() and (dest / "dishes.xlsx").exists()

    if name == "nutrition5k_side":
        return any("side" in d.name.lower() for d in dest.iterdir() if d.is_dir())

    return False

def extract_all(dest: Path):
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(dest)

for name, slug in kaggle_datasets.items():
    dest = BASE_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {name}")

    # Check if already extracted properly
    if expected_ok(name, dest):
        print(f"{name} already available.")
        continue

    # Check if zip exists
    zip_exists = any(f.suffix == ".zip" for f in dest.iterdir() if f.is_file())

    # Download if needed
    if not zip_exists:
        print(f"Downloading {name}...")
        !kaggle datasets download -d {slug} -p "{dest}"

    # Extract zip files
    print(f"Extracting {name}...")
    extract_all(dest)

    # Validate
    if expected_ok(name, dest):
        print(f"{name} extracted successfully.")
    else:
        print(f"{name} extraction incomplete. Manual review required.")

```

    
    Processing: foodseg103
    Downloading foodseg103...
    Dataset URL: https://www.kaggle.com/datasets/ggrill/foodseg103
    License(s): apache-2.0
    Downloading foodseg103.zip to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103
     99% 1.16G/1.17G [00:10<00:00, 54.6MB/s]
    100% 1.17G/1.17G [00:10<00:00, 117MB/s] 
    Extracting foodseg103...
    foodseg103 extracted successfully.
    
    Processing: nutrition5k
    Downloading nutrition5k...
    Dataset URL: https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset
    License(s): CC-BY-SA-4.0
    Downloading nutrition5k-dataset.zip to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k
    100% 2.29G/2.29G [00:19<00:00, 100MB/s] 
    100% 2.29G/2.29G [00:19<00:00, 123MB/s]
    Extracting nutrition5k...
    nutrition5k extracted successfully.
    
    Processing: nutrition5k_side
    Downloading nutrition5k_side...
    Dataset URL: https://www.kaggle.com/datasets/zygmuntyt/nutrition5k-dataset-side-angle-images
    License(s): unknown
    ^C
    Extracting nutrition5k_side...
    nutrition5k_side extraction incomplete. Manual review required.



```python
# ============================================================
# CELL 6 — Show Folder Structure for Quick Inspection
# ============================================================
def show_tree(path: Path, depth=0, max_depth=3):
    if not path.exists():
        print(f"{path} does not exist — skipping.")
        return
    if depth > max_depth:
        return
    prefix = "    " * depth
    print(f"{prefix}{path.name}/")
    for p in sorted(path.iterdir()):
        if p.is_dir():
            show_tree(p, depth+1, max_depth)
        else:
            print(f"{prefix}    {p.name}")

print("\n===== FoodSeg103 =====")
show_tree(BASE_DIR / "foodseg103")

print("\n===== Nutrition5k Original =====")
show_tree(BASE_DIR / "nutrition5k_orig")   # Now safe — will not crash

print("\n===== Nutrition5k SIDE-ANGLE =====")
show_tree(BASE_DIR / "nutrition5k_side")

```

    
    ===== FoodSeg103 =====
    foodseg103/
        FoodSeg103/
            ImageSets/
                test.txt
                train.txt
            Images/
                ann_dir/
                img_dir/
            Readme.txt
            category_id.txt
            test_recipe1m_id.txt
            train_test_recipe1m_id.txt
        foodseg103.zip
    
    ===== Nutrition5k Original =====
    /content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k_orig does not exist — skipping.
    
    ===== Nutrition5k SIDE-ANGLE =====
    nutrition5k_side/



```python
# Show a clean folder tree. Prints directories recursively and top-level files once.
from pathlib import Path

def show_tree(base_path, max_depth=6, show_top_files=True):
    base = Path(base_path).resolve()
    visited = set()

    def walk(dir_path: Path, depth: int):
        rp = dir_path.resolve()
        if rp in visited or depth > max_depth:
            return
        visited.add(rp)

        indent = ' ' * 4 * depth
        print(f"{indent}{dir_path.name}/")

        if depth == 0 and show_top_files:
            for f in sorted(p.name for p in dir_path.iterdir() if p.is_file()):
                print(f"{indent}    {f}")

        for d in sorted(p for p in dir_path.iterdir() if p.is_dir()):
            walk(d, depth + 1)

    print(f"\nDataset Structure for: {base.name}\n" + "-" * 80)
    walk(base, 0)
    print("-" * 80 + f"\nCompleted structure listing for {base.name}\n")

# Run for both datasets
show_tree(BASE_DIR / "foodseg103")
show_tree(BASE_DIR / "nutrition5k")

```

    
    Dataset Structure for: foodseg103
    --------------------------------------------------------------------------------
    foodseg103/
        foodseg103.zip
        FoodSeg103/
            ImageSets/
            Images/
                ann_dir/
                    test/
                    train/
                img_dir/
                    test/
                    train/
    --------------------------------------------------------------------------------
    Completed structure listing for foodseg103
    
    
    Dataset Structure for: nutrition5k
    --------------------------------------------------------------------------------
    nutrition5k/
        dish_images.pkl
        dish_ingredients.xlsx
        dishes.xlsx
        ingredients.xlsx
        nutrition5k-dataset.zip
    --------------------------------------------------------------------------------
    Completed structure listing for nutrition5k
    



```python
# Verify that all required dataset folders and key files exist.
from pathlib import Path

print("Verifying dataset completeness...\n")

foodseg_path = BASE_DIR / "foodseg103" / "FoodSeg103"
img_train = foodseg_path / "Images" / "img_dir" / "train"
ann_train = foodseg_path / "Images" / "ann_dir" / "train"

if foodseg_path.exists():
    print("FoodSeg103 folder found.")
    print(f"Train images: {len(list(img_train.glob('*.*')))} files")
    print(f"Train annotations: {len(list(ann_train.glob('*.*')))} files")
else:
    print("FoodSeg103 folder missing!")

nutrition_path = BASE_DIR / "nutrition5k"
required_files = ["dish_images.pkl", "dishes.xlsx", "ingredients.xlsx"]

print("\nNutrition5k dataset check:")
if nutrition_path.exists():
    print("Nutrition5k folder found.")
    for f in required_files:
        status = "Available" if (nutrition_path / f).exists() else "Missing"
        print(f"{f}: {status}")
else:
    print("Nutrition5k folder missing!")

print("\nDataset completeness verification finished.")

```

    Verifying dataset completeness...
    
    FoodSeg103 folder found.
    Train images: 4983 files
    Train annotations: 4983 files
    
    Nutrition5k dataset check:
    Nutrition5k folder found.
    dish_images.pkl: Available
    dishes.xlsx: Available
    ingredients.xlsx: Available
    
    Dataset completeness verification finished.



```python
import os, zipfile, pickle
from pathlib import Path

nutrition_dir = BASE_DIR / "nutrition5k"
zip_path = nutrition_dir / "nutrition5k-dataset.zip"
pkl_path = nutrition_dir / "dish_images.pkl"

print(f"Nutrition5k folder: {nutrition_dir.exists()}")
print(f"ZIP size: {zip_path.stat().st_size/1e6:.2f} MB" if zip_path.exists() else "ZIP missing")

if pkl_path.exists():
    bk = nutrition_dir / "dish_images_old.pkl"
    pkl_path.rename(bk)
    print(f"Backed up old PKL to {bk}")

extract_dir = nutrition_dir / "nutrition5k-dataset"
if extract_dir.exists():
    print("Existing extract folder found — skipping re-extract.")
else:
    print("Extracting Nutrition5k ZIP — this may take 2-3 minutes …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(nutrition_dir)
    print("Extraction complete.")

inner_pkl = None
for root, _, files in os.walk(extract_dir):
    for f in files:
        if f == "dish_images.pkl":
            inner_pkl = Path(root) / f
            break

if inner_pkl and inner_pkl.exists():
    print("Found inner PKL:", inner_pkl)
    with open(inner_pkl, "rb") as f:
        dish_images = pickle.load(f)
    import shutil
    shutil.copy(inner_pkl, pkl_path)
    print(f"Restored dish_images.pkl with depth images to {pkl_path}")
else:
    print("Could not find inner dish_images.pkl after extraction — manual inspection needed.")

```

    Nutrition5k folder: True
    ZIP size: 2462.23 MB
    Backed up old PKL to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k/dish_images_old.pkl
    Extracting Nutrition5k ZIP — this may take 2-3 minutes …
    Extraction complete.
    Could not find inner dish_images.pkl after extraction — manual inspection needed.



```python
import zipfile, os
from pathlib import Path

zip_path = Path("/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k/nutrition5k-dataset.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    print("Total files in ZIP:", len(z.namelist()))
    print("First 30 entries:")
    for n in z.namelist()[:30]:
        print("  ", n)

```

    Total files in ZIP: 4
    First 30 entries:
       dish_images.pkl
       dish_ingredients.xlsx
       dishes.xlsx
       ingredients.xlsx



```python
import pickle
import pandas as pd
from pathlib import Path

pkl_path = Path("/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k/dish_images.pkl")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
if isinstance(data, pd.DataFrame):
    print("Columns:", list(data.columns))
elif isinstance(data, list):
    print("List length:", len(data), "| sample type:", type(data[0]))
elif isinstance(data, dict):
    print("Keys:", list(data.keys())[:5])

```

    Type: <class 'pandas.core.frame.DataFrame'>
    Columns: ['dish', 'rgb_image', 'depth_image']



```python
# Preview Nutrition5k metadata from dishes.xlsx.
import pandas as pd

meta_file = BASE_DIR / "nutrition5k" / "dishes.xlsx"

if meta_file.exists():
    df_meta = pd.read_excel(meta_file)
    print(f"Loaded Nutrition5k metadata: {df_meta.shape[0]} records")
    print("Columns:", list(df_meta.columns))
    display(df_meta.head())
else:
    print("dishes.xlsx not found in Nutrition5k folder.")

```

    Loaded Nutrition5k metadata: 5006 records
    Columns: ['dish_id', 'total_mass', 'total_calories', 'total_fat', 'total_carb', 'total_protein']




  <div id="df-89718192-b16f-4e73-90a6-2b5ef9f28592" class="colab-df-container">
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
      <th>total_mass</th>
      <th>total_calories</th>
      <th>total_fat</th>
      <th>total_carb</th>
      <th>total_protein</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dish_1561662216</td>
      <td>300.794281</td>
      <td>193</td>
      <td>12.387489</td>
      <td>28.218290</td>
      <td>18.633970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dish_1562688426</td>
      <td>137.569992</td>
      <td>88</td>
      <td>8.256000</td>
      <td>5.190000</td>
      <td>10.297000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dish_1561662054</td>
      <td>419.438782</td>
      <td>292</td>
      <td>23.838249</td>
      <td>26.351543</td>
      <td>25.910593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dish_1562008979</td>
      <td>382.936646</td>
      <td>290</td>
      <td>22.224644</td>
      <td>10.173570</td>
      <td>35.345387</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dish_1560455030</td>
      <td>20.590000</td>
      <td>103</td>
      <td>0.148000</td>
      <td>4.625000</td>
      <td>0.956000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-89718192-b16f-4e73-90a6-2b5ef9f28592')"
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
        document.querySelector('#df-89718192-b16f-4e73-90a6-2b5ef9f28592 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-89718192-b16f-4e73-90a6-2b5ef9f28592');
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


    <div id="df-4b61fc28-7402-4a6c-8a6c-d15a933aa4ee">
      <button class="colab-df-quickchart" onclick="quickchart('df-4b61fc28-7402-4a6c-8a6c-d15a933aa4ee')"
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
            document.querySelector('#df-4b61fc28-7402-4a6c-8a6c-d15a933aa4ee button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# Ensure dish_images.pkl exists and load it safely
from pathlib import Path
import pickle

pkl_path = Path("/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k/dish_images.pkl")

if pkl_path.exists():
    print("Loading dish_images.pkl ...")
    with open(pkl_path, "rb") as f:
        dish_images = pickle.load(f)
    print("Loaded successfully:", type(dish_images))
else:
    raise FileNotFoundError(f"dish_images.pkl NOT found at {pkl_path}")

```

    Loading dish_images.pkl ...
    Loaded successfully: <class 'pandas.core.frame.DataFrame'>



```python
#  Correct Nutrition5k visualization — handles PNG bytes safely
import pickle, io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("dish_images type:", type(dish_images))

if isinstance(dish_images, list):
    print(f"List detected with {len(dish_images)} entries")
    sample = dish_images[0]
    if isinstance(sample, dict) and "rgb_image" in sample:
        row0 = sample
    else:
        raise ValueError("Unexpected list structure — first item has no 'rgb_image'.")
elif hasattr(dish_images, "iloc"):
    row0 = dish_images.iloc[0]
else:
    raise TypeError("Unsupported data type for dish_images:", type(dish_images))

if "rgb_image" in row0:
    rgb_bytes = row0["rgb_image"]
    if isinstance(rgb_bytes, (bytes, bytearray)):
        rgb_image = Image.open(io.BytesIO(rgb_bytes))
    else:
        rgb_image = Image.fromarray(np.uint8(rgb_bytes))
    plt.figure(figsize=(4,4))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Nutrition5k — RGB Image")
    plt.show()

if "depth_image" in row0:
    depth_bytes = row0["depth_image"]
    if isinstance(depth_bytes, (bytes, bytearray)):
        depth_image = Image.open(io.BytesIO(depth_bytes))
        plt.figure(figsize=(4,4))
        plt.imshow(depth_image, cmap="gray")
        plt.axis("off")
        plt.title("Nutrition5k — Depth Image")
        plt.show()

```

    dish_images type: <class 'pandas.core.frame.DataFrame'>



    
![png](01_data_collection_and_management_new_files/01_data_collection_and_management_new_14_1.png)
    



    
![png](01_data_collection_and_management_new_files/01_data_collection_and_management_new_14_2.png)
    



```python
# Decode and visualize one RGB and depth image from Nutrition5k
from PIL import Image
import io

sample_row = dish_images.iloc[0]
print("Dish ID:", sample_row["dish"])

rgb_bytes = sample_row["rgb_image"]
rgb_image = Image.open(io.BytesIO(rgb_bytes))

depth_bytes = sample_row["depth_image"]
depth_image = Image.open(io.BytesIO(depth_bytes))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(rgb_image)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(depth_image, cmap="gray")
plt.title("Depth Image")
plt.axis("off")

plt.tight_layout()
plt.show()

```

    Dish ID: dish_1559243887



    
![png](01_data_collection_and_management_new_files/01_data_collection_and_management_new_15_1.png)
    



```python
# Preview a sample image and its segmentation mask from FoodSeg103
import random
from PIL import Image
import matplotlib.pyplot as plt

img_train_dir = BASE_DIR / "foodseg103" / "FoodSeg103" / "Images" / "img_dir" / "train"
ann_train_dir = BASE_DIR / "foodseg103" / "FoodSeg103" / "Images" / "ann_dir" / "train"

sample_image_path = random.choice(list(img_train_dir.glob("*.jpg")))
sample_mask_path = ann_train_dir / (sample_image_path.stem + ".png")

rgb_img = Image.open(sample_image_path).convert("RGB")
mask_img = Image.open(sample_mask_path)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("FoodSeg103 - RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_img, cmap="nipy_spectral")
plt.title("FoodSeg103 - Segmentation Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Displayed sample: {sample_image_path.name}")

```


    
![png](01_data_collection_and_management_new_files/01_data_collection_and_management_new_16_0.png)
    


    Displayed sample: 00001797.jpg



```python
# Create a dataset manifest JSON file containing paths to all datasets.
import json

manifest = {
    "FoodSeg103": str(BASE_DIR / "foodseg103" / "FoodSeg103"),
    "Nutrition5k": str(BASE_DIR / "nutrition5k")
}

manifest_dir = BASE_DIR.parent / "manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)
manifest_path = manifest_dir / "datasets_manifest.json"

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)

print("Dataset manifest created successfully:")
print(json.dumps(manifest, indent=4))

```

    Dataset manifest created successfully:
    {
        "FoodSeg103": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103",
        "Nutrition5k": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k"
    }



```python
# Verify that manifest paths are correct and summarize basic dataset statistics.
import json
from pathlib import Path

with open(manifest_path, "r") as f:
    manifest_data = json.load(f)

print("Manifest contents:")
for name, path in manifest_data.items():
    print(f" - {name}: {path}")

foodseg_train = Path(manifest_data["FoodSeg103"]) / "Images" / "img_dir" / "train"
foodseg_count = len(list(foodseg_train.glob('*.*'))) if foodseg_train.exists() else 0

nutrition_files = len(list(Path(manifest_data["Nutrition5k"]).glob('*.*')))

print(f"\nSummary:")
print(f"FoodSeg103 - Training images: {foodseg_count}")
print(f"Nutrition5k - Total files: {nutrition_files}")

```

    Manifest contents:
     - FoodSeg103: /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103
     - Nutrition5k: /content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k
    
    Summary:
    FoodSeg103 - Training images: 4983
    Nutrition5k - Total files: 6



```python
# Upload USDA FoodData Central CSV manually.
from google.colab import files
import shutil

print("Upload your USDA CSV file (for example: food.csv, nutrient.csv)")
uploaded = files.upload()

usda_dir = BASE_DIR / "usda"
usda_dir.mkdir(parents=True, exist_ok=True)

for filename in uploaded.keys():
    src = Path(filename)
    dst = usda_dir / src.name
    shutil.move(str(src), dst)
    print(f"Moved {filename} to {dst}")

```

    Upload your USDA CSV file (for example: food.csv, nutrient.csv)




     <input type="file" id="files-aabeb8c9-c46d-421d-858c-a4897d3b0a9e" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-aabeb8c9-c46d-421d-858c-a4897d3b0a9e">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving acquisition_samples.csv to acquisition_samples.csv
    Saving agricultural_samples.csv to agricultural_samples.csv
    Saving food_attribute_type.csv to food_attribute_type.csv
    Saving food_attribute.csv to food_attribute.csv
    Saving food_calorie_conversion_factor.csv to food_calorie_conversion_factor.csv
    Saving food_category.csv to food_category.csv
    Saving food_component.csv to food_component.csv
    Saving food_nutrient_conversion_factor.csv to food_nutrient_conversion_factor.csv
    Saving food_nutrient.csv to food_nutrient.csv
    Saving food_portion.csv to food_portion.csv
    Saving food_protein_conversion_factor.csv to food_protein_conversion_factor.csv
    Saving food_update_log_entry.csv to food_update_log_entry.csv
    Saving food.csv to food.csv
    Saving foundation_food.csv to foundation_food.csv
    Saving input_food.csv to input_food.csv
    Saving kaggle.json to kaggle.json
    Saving lab_method_code.csv to lab_method_code.csv
    Saving lab_method_nutrient.csv to lab_method_nutrient.csv
    Saving lab_method.csv to lab_method.csv
    Saving market_acquisition.csv to market_acquisition.csv
    Saving measure_unit.csv to measure_unit.csv
    Saving nutrient.csv to nutrient.csv
    Saving sample_food.csv to sample_food.csv
    Saving sub_sample_food.csv to sub_sample_food.csv
    Saving sub_sample_result.csv to sub_sample_result.csv
    Moved acquisition_samples.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/acquisition_samples.csv
    Moved agricultural_samples.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/agricultural_samples.csv
    Moved food_attribute_type.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_attribute_type.csv
    Moved food_attribute.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_attribute.csv
    Moved food_calorie_conversion_factor.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_calorie_conversion_factor.csv
    Moved food_category.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_category.csv
    Moved food_component.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_component.csv
    Moved food_nutrient_conversion_factor.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_nutrient_conversion_factor.csv
    Moved food_nutrient.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_nutrient.csv
    Moved food_portion.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_portion.csv
    Moved food_protein_conversion_factor.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_protein_conversion_factor.csv
    Moved food_update_log_entry.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food_update_log_entry.csv
    Moved food.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/food.csv
    Moved foundation_food.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/foundation_food.csv
    Moved input_food.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/input_food.csv
    Moved kaggle.json to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/kaggle.json
    Moved lab_method_code.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/lab_method_code.csv
    Moved lab_method_nutrient.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/lab_method_nutrient.csv
    Moved lab_method.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/lab_method.csv
    Moved market_acquisition.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/market_acquisition.csv
    Moved measure_unit.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/measure_unit.csv
    Moved nutrient.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/nutrient.csv
    Moved sample_food.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/sample_food.csv
    Moved sub_sample_food.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/sub_sample_food.csv
    Moved sub_sample_result.csv to /content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda/sub_sample_result.csv



```python
# Verify USDA CSV files and preview contents
import pandas as pd

usda_dir = BASE_DIR / "usda"
csv_files = sorted([f.name for f in usda_dir.glob("*.csv")])
print(f"Found {len(csv_files)} USDA CSV files:")
for f in csv_files:
    print(" -", f)

food_csv = usda_dir / "food.csv"
nutrient_csv = usda_dir / "nutrient.csv"

if food_csv.exists():
    df_food = pd.read_csv(food_csv, nrows=5)
    print("\nPreview of food.csv:")
    display(df_food.head())
else:
    print("\nfood.csv not found.")

if nutrient_csv.exists():
    df_nutrient = pd.read_csv(nutrient_csv, nrows=5)
    print("\nPreview of nutrient.csv:")
    display(df_nutrient.head())
else:
    print("\nnutrient.csv not found.")

```

    Found 24 USDA CSV files:
     - acquisition_samples.csv
     - agricultural_samples.csv
     - food.csv
     - food_attribute.csv
     - food_attribute_type.csv
     - food_calorie_conversion_factor.csv
     - food_category.csv
     - food_component.csv
     - food_nutrient.csv
     - food_nutrient_conversion_factor.csv
     - food_portion.csv
     - food_protein_conversion_factor.csv
     - food_update_log_entry.csv
     - foundation_food.csv
     - input_food.csv
     - lab_method.csv
     - lab_method_code.csv
     - lab_method_nutrient.csv
     - market_acquisition.csv
     - measure_unit.csv
     - nutrient.csv
     - sample_food.csv
     - sub_sample_food.csv
     - sub_sample_result.csv
    
    Preview of food.csv:




  <div id="df-8b7c6fbc-13f9-4605-816c-d1fdc4dd07ec" class="colab-df-container">
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
      <th>data_type</th>
      <th>description</th>
      <th>food_category_id</th>
      <th>publication_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>319874</td>
      <td>sample_food</td>
      <td>HUMMUS, SABRA CLASSIC</td>
      <td>16</td>
      <td>2019-04-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>319875</td>
      <td>market_acquisition</td>
      <td>HUMMUS, SABRA CLASSIC</td>
      <td>16</td>
      <td>2019-04-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>319876</td>
      <td>market_acquisition</td>
      <td>HUMMUS, SABRA CLASSIC</td>
      <td>16</td>
      <td>2019-04-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319877</td>
      <td>sub_sample_food</td>
      <td>Hummus</td>
      <td>16</td>
      <td>2019-04-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>319878</td>
      <td>sub_sample_food</td>
      <td>Hummus</td>
      <td>16</td>
      <td>2019-04-01</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8b7c6fbc-13f9-4605-816c-d1fdc4dd07ec')"
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
        document.querySelector('#df-8b7c6fbc-13f9-4605-816c-d1fdc4dd07ec button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8b7c6fbc-13f9-4605-816c-d1fdc4dd07ec');
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


    <div id="df-880e498f-ddd6-4e2e-b564-d2aaee662ce3">
      <button class="colab-df-quickchart" onclick="quickchart('df-880e498f-ddd6-4e2e-b564-d2aaee662ce3')"
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
            document.querySelector('#df-880e498f-ddd6-4e2e-b564-d2aaee662ce3 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    
    Preview of nutrient.csv:




  <div id="df-2faa7f55-9ee6-4886-b69a-e19b24dd84b3" class="colab-df-container">
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
      <th>id</th>
      <th>name</th>
      <th>unit_name</th>
      <th>nutrient_nbr</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2047</td>
      <td>Energy (Atwater General Factors)</td>
      <td>KCAL</td>
      <td>957</td>
      <td>280.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2048</td>
      <td>Energy (Atwater Specific Factors)</td>
      <td>KCAL</td>
      <td>958</td>
      <td>290.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>Solids</td>
      <td>G</td>
      <td>201</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002</td>
      <td>Nitrogen</td>
      <td>G</td>
      <td>202</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1003</td>
      <td>Protein</td>
      <td>G</td>
      <td>203</td>
      <td>600.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2faa7f55-9ee6-4886-b69a-e19b24dd84b3')"
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
        document.querySelector('#df-2faa7f55-9ee6-4886-b69a-e19b24dd84b3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2faa7f55-9ee6-4886-b69a-e19b24dd84b3');
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


    <div id="df-8dd2c216-c26b-4622-82f3-2fe4336820dc">
      <button class="colab-df-quickchart" onclick="quickchart('df-8dd2c216-c26b-4622-82f3-2fe4336820dc')"
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
            document.querySelector('#df-8dd2c216-c26b-4622-82f3-2fe4336820dc button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# Add USDA dataset path to manifest
import json

manifest_path = manifest_dir / "datasets_manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

manifest["USDA_FoodData"] = str(usda_dir)

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)

print("Updated dataset manifest with USDA path:")
print(json.dumps(manifest, indent=4))

```

    Updated dataset manifest with USDA path:
    {
        "FoodSeg103": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103",
        "Nutrition5k": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/nutrition5k",
        "USDA_FoodData": "/content/drive/MyDrive/Cal_Estimation_Project/data/raw/usda"
    }



```python
# ============================================================
# SAVE CACHE FOR NOTEBOOK 01 (NEW)
# Saves lightweight manifest + dataset paths for fast reuse.
# ============================================================

cache_data = {
    "manifest": manifest,
    "foodseg_path": manifest["FoodSeg103"],
    "nutrition5k_path": manifest["Nutrition5k"],
    "usda_path": manifest["USDA_FoodData"]
}

save_cache(cache_data, "01_cache")

```

    ✔ Saved cache → /content/drive/MyDrive/Cal_Estimation_Project/cache/01_cache.pkl



```python

```
