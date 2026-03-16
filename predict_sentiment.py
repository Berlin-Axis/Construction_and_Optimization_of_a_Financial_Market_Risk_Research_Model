"""
用FinBERT预测未标注数据的情感
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import os

def load_model():
    """加载训练好的模型"""
    print("加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model_path = './finbert-finetuned'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

def predict_sentiment(texts, tokenizer, model, device, batch_size=32):
    """批量预测情感"""
    sentiments = []
    confidence_scores = []
    
    sentiment_mapping = {0: '消极', 1: '中性', 2: '积极'}
    
    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="预测中"):
        batch_texts = texts[i:i+batch_size]
        
        # 编码文本
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        # 保存结果
        for pred, prob in zip(preds.cpu().numpy(), probs.cpu().numpy()):
            sentiments.append(sentiment_mapping[pred])
            confidence_scores.append(float(prob[pred]))
    
    return sentiments, confidence_scores

def main():
    """主函数"""
    print("=" * 60)
    print("FinBERT 情感预测")
    print("=" * 60)
    
    # 1. 加载模型
    tokenizer, model, device = load_model()
    
    # 2. 读取未标注数据
    print("\n读取未标注数据...")
    df = pd.read_csv('data/cleaned_data/清洗后的全部评论.csv', encoding='utf-8-sig')
    print(f"未标注数据量: {len(df)} 条")
    
    # 3. 预测情感
    print("\n开始预测...")
    texts = df['清洗后评论'].tolist()
    sentiments, confidence_scores = predict_sentiment(texts, tokenizer, model, device)
    
    # 4. 保存预测结果
    df['预测情感'] = sentiments
    df['置信度'] = confidence_scores
    
    os.makedirs('data/predicted_data', exist_ok=True)
    output_file = 'data/predicted_data/情感预测结果.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {output_file}")
    
    # 5. 随机抽查20条
    print("\n" + "=" * 60)
    print("随机抽查 20 条预测结果")
    print("=" * 60)
    
    sample_df = df.sample(n=20, random_state=42)
    
    print("\n" + "-" * 60)
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\n【样本 {i}/20】")
        print(f"股票: {row['股票名称']}")
        print(f"评论: {row['清洗后评论']}")
        print(f"预测情感: {row['预测情感']}")
        print(f"置信度: {row['置信度']:.4f}")
        print("-" * 60)
    
    # 6. 统计预测结果
    print("\n" + "=" * 60)
    print("预测结果统计")
    print("=" * 60)
    
    sentiment_counts = df['预测情感'].value_counts(normalize=True)
    print("\n情感分布:")
    for sentiment, proportion in sentiment_counts.items():
        count = df['预测情感'].value_counts()[sentiment]
        print(f"  {sentiment}: {count} 条 ({proportion*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()