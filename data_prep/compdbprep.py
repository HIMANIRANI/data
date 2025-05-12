import os
import json
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cuda'}
)

# 2. Prepare documents
documents = []
company_folder = 'updated_company_data'  # Adjust your folder path if needed

# Dictionary of how many days needed for each indicator
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
            
            # Building page content
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
            
            # Add available indicators
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
            
            page_content = "\n".join(lines).strip()
            documents.append(Document(page_content=page_content))

print(f"✅ Prepared {len(documents)} company documents!")

# 3. Build FAISS index
company_vectorstore = FAISS.from_documents(documents, embedding_model)

# 4. Save FAISS index
save_path = 'company_vec'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
company_vectorstore.save_local(save_path)

print(f"✅ Company FAISS vectorstore saved at {save_path}")
