import requests
from bs4 import BeautifulSoup
import pandas as pd
from torch.nn.functional import cosine_similarity
from transformers import BertModel, DistilBertModel
bert_model = BertModel.from_pretrained('monologg/kobert')
distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
from tokenization_kobert import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
import torch
from kobert_transformers import get_kobert_model, get_distilkobert_model
model = get_kobert_model()
model.eval()
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
sequence_output[0]

def get_vector(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**tokens)
    if outputs.last_hidden_state.size(1) > 0:  # 첫 번째 차원의 크기 확인
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
    else:
        # 적절한 오류 처리 또는 대체 로직
        print("Error: No output from model")
        return None
def scrape_esg_companies(base_url, start_page=1, end_page=50):
    all_data = []
    for page in range(start_page, end_page + 1):
        url = f"{base_url}?pg={page}&block=2"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='UTF-8')

        rows = soup.find_all('tr')
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) > 5:  # ESG 수준이 6번째 열에 있다고 가정
                all_data.append({
                    'name': cols[1].text.strip(),
                    'industry': cols[2].text.strip(),
                    'main_product': cols[3].text.strip(),
                    'esg_level': cols[5].text.strip(),  # ESG 수준을 가져오는 부분 수정
                })
    return pd.DataFrame(all_data)

def recommend_companies(user_input, df):
    user_input_vector = get_vector(user_input)
    
    scores = []
    for index, row in df.iterrows():
        combined_info = row['industry'] + " " + row['main_product']
        combined_info_vector = get_vector(combined_info)
        sim_score_tensor = (1 + cosine_similarity(user_input_vector, combined_info_vector)) / 2 * 100
        sim_score = sim_score_tensor.item()  # 스칼라 값으로 변환
        scores.append((index, sim_score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [(df['name'].iloc[i[0]], i[1]) for i in scores[:4]]
    
    return recommendations

if __name__ == "__main__":
    base_url = "https://www.esgsupport.or.kr/sub_news/list.php"
    df = scrape_esg_companies(base_url)  
    
    user_input = input("키워드나 조건을 입력하세요: ")
    recommendations = recommend_companies(user_input, df)


    backend_data = []

    for company_name, score in recommendations:
        backend_data.append({
            'company_name': company_name,
            'similarity_score': score
        })

