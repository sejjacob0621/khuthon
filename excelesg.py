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
        print("Error: No output from model")
        return None

def load_esg_companies_from_excel(file_path):
    # 엑셀 파일을 DataFrame으로 로드하고 필요한 열만 남김
    # header=2를 사용하여 첫 2행을 건너뛰고 3행을 헤더(열 이름)로 사용
    df = pd.read_excel(file_path, header=2)
    df = df[['단체명', '주된사업']]  # 필요한 열만 남김
    return df

def recommend_companies(user_input, df):
    user_input_vector = get_vector(user_input)
    
    scores = []
    for index, row in df.iterrows():
        main_activity_vector = get_vector(row['주된사업'])
        # 유사도 점수를 백분율로 변환
        sim_score_tensor = (1+cosine_similarity(user_input_vector, main_activity_vector)) /2 * 100
        sim_score = sim_score_tensor.item()
        scores.append((row['단체명'], sim_score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = scores[:4]
    
    return recommendations

if __name__ == "__main__":
    file_path = "환경부 등록 비영리민간단체 현황(2021.1.7).xls"  
    df = load_esg_companies_from_excel(file_path)  
    
    user_input = input("키워드나 조건을 입력하세요: ")
    recommendations = recommend_companies(user_input, df)
    print("추천하는 ESG 기업들:")
    for i, (company, score) in enumerate(recommendations, 1):
        print(f"{i}번: {company} (유사도: {score:.2f}%)")
