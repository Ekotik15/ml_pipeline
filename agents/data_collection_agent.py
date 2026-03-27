import os
import pandas as pd
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

    # Методы scrape и fetch_api удалены – они не используются в пайплайне
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
