from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent()
sources = [{"type": "hf_dataset", "name": "imdb"}]
df = agent.run(sources)
print("Готово! CSV-файл сохранён в data/raw/")
