"""
FinBERT 微调金融情感分类
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

class FinancialDataset(Dataset):
    """金融数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def load_data():
    """加载训练数据"""
    train_df = pd.read_csv('data/train_data/训练集.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('data/train_data/验证集.csv', encoding='utf-8-sig')
    
    return train_df, val_df

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def finetune():
    """微调FinBERT"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    model_name = 'valuesimplex-ai-lab/finbert2-base'
    print(f"加载模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    # 加载数据
    train_df, val_df = load_data()
    
    # 创建数据集
    train_dataset = FinancialDataset(train_df['清洗后评论'].tolist(), train_df['情感'].tolist(), tokenizer)
    val_dataset = FinancialDataset(val_df['清洗后评论'].tolist(), val_df['情感'].tolist(), tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 设置优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练参数
    num_epochs = 3
    best_accuracy = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        val_accuracy = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained('./finbert-finetuned')
            tokenizer.save_pretrained('./finbert-finetuned')
            print(f'保存最佳模型, 准确率: {best_accuracy:.4f}')
    
    print("\n" + "=" * 50)
    print("FinBERT 微调完成!")
    print(f"最佳验证集准确率: {best_accuracy:.4f}")
    print("模型已保存到 ./finbert-finetuned")
    print("=" * 50)

if __name__ == "__main__":
    finetune()