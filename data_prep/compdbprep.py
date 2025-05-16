import os
import json
import torch
import pickle
import gzip
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the embedding model
model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cuda'}
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Define paths
company_folder = 'updated_company_data'
save_path = 'company_vec'
texts_cache = 'texts.pkl.gz'
embeddings_cache = 'embeddings.pkl.gz'

# 3. Safe pickle load/save functions using gzip
def safe_load_pickle(filename):
    if not os.path.exists(filename):
        return []
    try:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return []

def safe_save_pickle(obj, filename):
    tmp_filename = filename + '.tmp'
    with gzip.open(tmp_filename, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_filename, filename)  # Atomic replace

# 4. Prepare documents
documents = []
indicator_requirements = {
    "SMA": 10,
    "EMA": 12,
    "RSI": 14,
    "MACD": 26,
    "Bollinger Bands": 20
}

for filename in os.listdir(company_folder):
    if filename.endswith('.json'):
        symbol = filename.replace('.json', '')
        file_path = os.path.join(company_folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            company_data = json.load(f)
        
        for date, data in company_data.items():
            price = data.get('price', {})
            indicators = data.get('indicators', {})
            
            lines = [
                f"Company Symbol: {symbol}",
                f"Date: {date}",
                f"Open Price: {price.get('open', 'N/A')}",
                f"Close Price: {price.get('close', 'N/A')}",
                f"Max Price: {price.get('max', 'N/A')}",
                f"Min Price: {price.get('min', 'N/A')}",
                f"Traded Shares: {data.get('tradedShares', 'N/A')}",
                f"Amount: {data.get('amount', 'N/A')}",
            ]
            
            for indicator_name, required_days in indicator_requirements.items():
                if indicator_name == "Bollinger Bands":
                    bb_high = indicators.get('BB_High')
                    bb_low = indicators.get('BB_Low')
                    bb_mid = indicators.get('BB_Mid')
                    if bb_high is not None and bb_low is not None and bb_mid is not None:
                        lines.append(f"Bollinger Bands High: {bb_high}")
                        lines.append(f"Bollinger Bands Low: {bb_low}")
                        lines.append(f"Bollinger Bands Mid: {bb_mid}")
                    else:
                        lines.append(f"Bollinger Bands not available: Requires minimum {required_days} days of data.")
                else:
                    value = indicators.get(indicator_name)
                    if value is not None:
                        lines.append(f"{indicator_name}: {value}")
                    else:
                        lines.append(f"{indicator_name} not available: Requires minimum {required_days} days of data.")

            documents.append(Document(page_content="\n".join(lines)))

print(f"✅ Total documents prepared: {len(documents)}")

# 5. Load cached embeddings if available
texts = safe_load_pickle(texts_cache)
embeddings = safe_load_pickle(embeddings_cache)

if texts and embeddings:
    print(f"📂 Found previous cache! Resuming from document {len(texts)}")
else:
    print("🆕 No previous cache found. Starting fresh...")
texts_to_embed = [doc.page_content for doc in documents[len(texts):]]

# 6. Embed in batches (save only once at the end)
batch_size = 32
for batch_start in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding in batches"):
    batch_texts = texts_to_embed[batch_start:batch_start + batch_size]
    try:
        batch_embeddings = model.embed_documents(batch_texts)
        texts.extend(batch_texts)
        embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"❌ Error in batch starting at document {len(texts) + 1}: {e}")
        break

# 7. Final save
safe_save_pickle(texts, texts_cache)
safe_save_pickle(embeddings, embeddings_cache)
print(f"💾 Final save after {len(texts)} total documents.")
print("✅ All embeddings completed and cached.")

# 8. Build FAISS vectorstore
print("📦 Building FAISS vectorstore...")
company_vectorstore = FAISS.from_embeddings(
    list(zip(texts, embeddings)),
    embedding=model
)

# 9. Save FAISS index
if not os.path.exists(save_path):
    os.makedirs(save_path)
company_vectorstore.save_local(save_path)
print(f"✅ Company FAISS vectorstore saved at {save_path}")
