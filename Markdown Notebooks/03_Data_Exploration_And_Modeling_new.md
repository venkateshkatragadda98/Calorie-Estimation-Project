```python
# Mount Google Drive and import required libraries
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Define base directories
BASE_DIR = Path('/content/drive/MyDrive/Cal_Estimation_Project/data')
PROCESSED_DIR = BASE_DIR / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

print("Environment setup complete and Drive mounted.")

```

    Mounted at /content/drive
    Environment setup complete and Drive mounted.



```python
# ============================================================
# CACHE UTILITIES FOR NOTEBOOK 03 (NEW)
# ============================================================

import pickle
import os

CACHE_DIR = "/content/drive/MyDrive/Cal_Estimation_Project/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(obj, name="03_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved cache → {path}")

def load_cache(name="03_cache"):
    path = f"{CACHE_DIR}/{name}.pkl"
    if os.path.exists(path):
        print(f"✔ Loaded cache → {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

```


```python
# Load processed datasets from Notebook 02
foodseg = np.load(PROCESSED_DIR / 'foodseg103_samples.npz', allow_pickle=True)
nutrition5k = np.load(PROCESSED_DIR / 'nutrition5k_samples.npz', allow_pickle=True)
usda = pd.read_csv(PROCESSED_DIR / 'usda_reference.csv')

print("Datasets loaded successfully:\n")
print(f"FoodSeg103 → Images: {len(foodseg['images'])}, Masks: {len(foodseg['masks'])}")
print(f"Nutrition5k → RGB: {len(nutrition5k['rgb'])}, Depth: {len(nutrition5k['depth'])}")
print(f"USDA → Rows: {usda.shape[0]}, Columns: {usda.shape[1]}")

```

    Datasets loaded successfully:
    
    FoodSeg103 → Images: 50, Masks: 50
    Nutrition5k → RGB: 5, Depth: 5
    USDA → Rows: 135, Columns: 6



```python
# Display a FoodSeg103 image and its segmentation mask
idx = random.randint(0, len(foodseg['images']) - 1)
img = foodseg['images'][idx]
mask = foodseg['masks'][idx]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("FoodSeg103 Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='nipy_spectral')
plt.title("Segmentation Mask")
plt.axis("off")
plt.show()

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_3_0.png)
    



```python
# Display a Nutrition5k RGB and Depth pair
idx = random.randint(0, len(nutrition5k['rgb']) - 1)
rgb = nutrition5k['rgb'][idx]
depth = nutrition5k['depth'][idx]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(rgb)
plt.title("Nutrition5k RGB Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(depth, cmap='gray')
plt.title("Depth Map (Food Portion Proxy)")
plt.axis("off")
plt.show()

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_4_0.png)
    



```python
# Visualize calorie distribution and show nutrient statistics
plt.figure(figsize=(7,4))
usda['calories_kcal'].dropna().hist(bins=50, color='orange', edgecolor='black')
plt.title("Calorie Distribution (USDA Reference Data)")
plt.xlabel("Calories per 100g")
plt.ylabel("Count")
plt.show()

print("📊 Nutrient Summary Statistics:")
print(usda[['calories_kcal', 'protein_g', 'fat_g', 'carbs_g']].describe())

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_5_0.png)
    


    📊 Nutrient Summary Statistics:
           calories_kcal   protein_g       fat_g     carbs_g
    count     135.000000  135.000000  135.000000  135.000000
    mean      194.149916   11.777185    9.373481   16.017111
    std       165.669981   12.589173   15.148370   23.864253
    min        11.975150    0.000000    0.000000    0.000000
    25%        52.540660    1.130000    0.340000    2.070000
    50%       147.733351    7.810000    2.300000    6.020000
    75%       326.621595   19.300000   11.500000   15.750000
    max       832.370440   79.900000   99.100000   99.600000



```python
# Scatterplots showing how calories relate to protein, fat, and carbs
fig, axes = plt.subplots(1,3,figsize=(15,4))

axes[0].scatter(usda['protein_g'], usda['calories_kcal'], alpha=0.5)
axes[0].set_title("Calories vs Protein")
axes[0].set_xlabel("Protein (g)")
axes[0].set_ylabel("Calories (kcal)")

axes[1].scatter(usda['fat_g'], usda['calories_kcal'], alpha=0.5, color='red')
axes[1].set_title("Calories vs Fat")
axes[1].set_xlabel("Fat (g)")

axes[2].scatter(usda['carbs_g'], usda['calories_kcal'], alpha=0.5, color='green')
axes[2].set_title("Calories vs Carbs")
axes[2].set_xlabel("Carbs (g)")

plt.tight_layout()
plt.show()

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_6_0.png)
    



```python
# Display correlation heatmap among nutrients
corr = usda[['calories_kcal', 'protein_g', 'fat_g', 'carbs_g']].corr()

plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix (Calories vs Nutrients)")
plt.show()

print("Correlation Values:")
print(corr)

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_7_0.png)
    


    Correlation Values:
                   calories_kcal  protein_g     fat_g   carbs_g
    calories_kcal       1.000000   0.522581  0.800938  0.347715
    protein_g           0.522581   1.000000  0.374535 -0.195509
    fat_g               0.800938   0.374535  1.000000 -0.189496
    carbs_g             0.347715  -0.195509 -0.189496  1.000000



```python
# Create feature and target variables for regression modeling
X = usda[['protein_g', 'fat_g', 'carbs_g']].fillna(0)
y = usda['calories_kcal'].fillna(0)

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

```

    Feature matrix shape: (135, 3)
    Target vector shape: (135,)



```python
# Train a baseline regression model using USDA nutrient data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Baseline Model Trained Successfully\n")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f} kcal per 100g")

```

    Baseline Model Trained Successfully
    
    R² Score: 0.997
    Mean Absolute Error: 6.56 kcal per 100g



```python
# Display learned coefficients for interpretability
coef_df = pd.DataFrame({
    'Nutrient': X.columns,
    'Coefficient (kcal per gram)': model.coef_.round(2)
}).sort_values('Coefficient (kcal per gram)', ascending=False)

print("Nutrient Contributions to Calorie Prediction:")
display(coef_df)

```

    Nutrient Contributions to Calorie Prediction:




  <div id="df-c85073ec-67f3-4b46-8850-c666a68481f6" class="colab-df-container">
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
      <th>Nutrient</th>
      <th>Coefficient (kcal per gram)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>fat_g</td>
      <td>8.51</td>
    </tr>
    <tr>
      <th>0</th>
      <td>protein_g</td>
      <td>4.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>carbs_g</td>
      <td>3.92</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c85073ec-67f3-4b46-8850-c666a68481f6')"
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
        document.querySelector('#df-c85073ec-67f3-4b46-8850-c666a68481f6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c85073ec-67f3-4b46-8850-c666a68481f6');
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


    <div id="df-f2de9710-4b82-4788-ba80-745004f09707">
      <button class="colab-df-quickchart" onclick="quickchart('df-f2de9710-4b82-4788-ba80-745004f09707')"
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
            document.querySelector('#df-f2de9710-4b82-4788-ba80-745004f09707 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_9c671ee4-9732-4652-b285-9ae84b6adb6e">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('coef_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_9c671ee4-9732-4652-b285-9ae84b6adb6e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('coef_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
# Compare predicted vs actual calorie values
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Calories (kcal)")
plt.ylabel("Predicted Calories (kcal)")
plt.title("Predicted vs Actual Calorie Values")
plt.show()

```


    
![png](03_Data_Exploration_And_Modeling_new_files/03_Data_Exploration_And_Modeling_new_11_0.png)
    



```python
# Save baseline model coefficients
coef_df.to_csv(OUTPUT_DIR / 'baseline_model_coefficients.csv', index=False)
print("💾 Baseline model coefficients saved to /outputs/baseline_model_coefficients.csv")

```

    💾 Baseline model coefficients saved to /outputs/baseline_model_coefficients.csv



```python
# ============================================================
# SAVE CACHE FOR NOTEBOOK 03 (NEW)
# ============================================================

cache_data = {
    "coef_table_path": str(OUTPUT_DIR / "baseline_model_coefficients.csv"),
    "num_foodseg_samples": len(foodseg['images']),
    "num_nutrition5k_samples": len(nutrition5k['rgb']),
    "usda_rows": usda.shape[0]
}

save_cache(cache_data, "03_cache")

```

    ✔ Saved cache → /content/drive/MyDrive/Cal_Estimation_Project/cache/03_cache.pkl

