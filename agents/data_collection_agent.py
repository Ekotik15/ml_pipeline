import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class DataCollectionAgent:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path

    def _unify(self, df, source_name, content_col="text", label_col="label"):
        if content_col not in df.columns:
            raise ValueError(f"Колонка {content_col} не найдена")
        if label_col not in df.columns:
            df[label_col] = None
        unified = pd.DataFrame()
        unified["content"] = df[content_col]
        unified["label"] = df[label_col]
        unified["source"] = source_name
        unified["collected_at"] = datetime.now().isoformat()
        return unified

    def load_dataset(self, name, source="hf"):
        if source == "hf":
            from datasets import load_dataset
            ds = load_dataset(name, split="train")
            df = ds.to_pandas()
            return self._unify(df, f"hf_{name}", "text", "label")
        elif source == "kaggle":
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                api.dataset_download_files(name, path=tmpdir, unzip=True)
                files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
                if not files:
                    raise FileNotFoundError("CSV файл не найден")
                df = pd.read_csv(os.path.join(tmpdir, files[0]))
            text_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) else df.columns[0]
            label_col = df.select_dtypes(include=['int64', 'float64']).columns[0] if len(df.select_dtypes(include=['int64', 'float64']).columns) else None
            return self._unify(df, f"kaggle_{name}", text_col, label_col)
        else:
            raise ValueError("source должен быть 'hf' или 'kaggle'")

    def scrape(self, url, selector):
        if not selector:
            raise ValueError("Необходимо указать selector")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.select(selector)
        if not elements:
            raise ValueError(f"По селектору '{selector}' ничего не найдено")
        data = [{"text": el.get_text(strip=True), "label": None} for el in elements if el.get_text(strip=True)]
        df = pd.DataFrame(data)
        return self._unify(df, f"scrape_{url}", "text", "label")

    def fetch_api(self, endpoint, params=None):
        params = params or {}
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "articles" in data:
            rows = []
            for art in data["articles"]:
                text = f"{art.get('title','')} {art.get('description','')}".strip()
                if text:
                    rows.append({"text": text, "label": None})
            df = pd.DataFrame(rows)
        else:
            df = pd.json_normalize(data)
            text_col = df.select_dtypes(include=['object']).columns[0] if not df.empty else "text"
            return self._unify(df, f"api_{endpoint}", text_col, None)
        return self._unify(df, f"api_{endpoint}", "text", "label")

    def merge(self, sources):
        dfs = []
        for src in sources:
            if isinstance(src, pd.DataFrame):
                dfs.append(src)
            elif isinstance(src, dict):
                if src["type"] == "hf_dataset":
                    df = self.load_dataset(src["name"], "hf")
                elif src["type"] == "kaggle_dataset":
                    df = self.load_dataset(src["name"], "kaggle")
                elif src["type"] == "scrape":
                    df = self.scrape(src["url"], src["selector"])
                elif src["type"] == "api":
                    df = self.fetch_api(src["endpoint"], src.get("params"))
                else:
                    raise ValueError(f"Неизвестный тип: {src['type']}")
                dfs.append(df)
            else:
                raise ValueError("Неверный формат источника")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def run(self, sources):
        df = self.merge(sources)
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/collected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Сохранено в {filename}")
        return df