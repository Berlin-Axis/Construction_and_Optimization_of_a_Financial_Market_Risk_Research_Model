import pandas as pd
import os
import re
import hashlib

script_dir = os.path.dirname(os.path.abspath(__file__))
comments_path = os.path.join(script_dir, 'data', 'comments')
output_path = os.path.join(script_dir, 'data', 'cleaned_data')

os.makedirs(output_path, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
    text = re.sub(r'[a-zA-Z]:\\[^\s]+', '', text)
    text = re.sub(r'[\uff01-\uff5e]', lambda x: chr(ord(x.group()) - 0xfee0), text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_meaningful(text, min_len=3, max_len=500):
    if not text or len(text) < min_len or len(text) > max_len:
        return False
    chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
    if chinese_count < 2:
        return False
    if re.match(r'^[0-9\s\.\,\+\-\*\/]+$', text):
        return False
    if re.match(r'^[\!\?\。\，\！\？\~\～\s]+$', text):
        return False
    return True

def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

print("=" * 60)
print("开始处理评论数据")
print("=" * 60)

all_data = []
stats = {}

for file in os.listdir(comments_path):
    if file.endswith('.xlsx'):
        stock_name = file.replace('.xlsx', '')
        file_path = os.path.join(comments_path, file)
        
        print(f"\n处理: {stock_name}")
        
        df = pd.read_excel(file_path)
        original_count = len(df)
        
        df['股票名称'] = stock_name
        
        info_mask = df['作者'].str.contains('资讯$', na=False)
        df = df[~info_mask]
        after_info = len(df)
        
        df['清洗后评论'] = df['评论内容'].apply(clean_text)
        df = df[df['清洗后评论'].apply(is_meaningful)]
        after_clean = len(df)
        
        df['内容哈希'] = df['清洗后评论'].apply(get_text_hash)
        df = df.drop_duplicates(subset='内容哈希', keep='first')
        after_dedup = len(df)
        
        stats[stock_name] = {
            '原始': original_count,
            '去资讯': after_info,
            '清洗后': after_clean,
            '去重后': after_dedup,
            '过滤率': f"{(1 - after_dedup/original_count)*100:.1f}%"
        }
        
        all_data.append(df)
        
        print(f"  原始: {original_count} → 去资讯: {after_info} → 清洗: {after_clean} → 去重: {after_dedup}")

print("\n" + "=" * 60)
print("各股票数据统计")
print("=" * 60)
stats_df = pd.DataFrame(stats).T
print(stats_df)

all_df = pd.concat(all_data, ignore_index=True)
print(f"\n\n清洗后总数据量: {len(all_df)} 条")

print("\n" + "=" * 60)
print("按股票分层抽样，准备标注数据...")
print("=" * 60)

sample_per_stock = min(100, len(all_df) // 8)
annotation_samples = []

for stock in all_df['股票名称'].unique():
    stock_df = all_df[all_df['股票名称'] == stock]
    if len(stock_df) > sample_per_stock:
        sample = stock_df.sample(n=sample_per_stock, random_state=42)
    else:
        sample = stock_df
    annotation_samples.append(sample)

annotation_df = pd.concat(annotation_samples, ignore_index=True)
annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

annotation_export = annotation_df[[
    '股票名称', '作者', '发布时间', '评论内容', '清洗后评论', '楼层'
]].copy()

annotation_export['情感(积极/中性/消极)'] = ''
annotation_export['事件类型'] = ''
annotation_export['备注'] = ''

cleaned_file = os.path.join(output_path, '清洗后的全部评论.csv')
annotation_file = os.path.join(output_path, '待标注评论.xlsx')
stats_file = os.path.join(output_path, '数据清洗统计.csv')

all_df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')
print(f"\n清洗后数据已保存: {cleaned_file}")
print(f"数据量: {len(all_df)} 条")

annotation_export.to_excel(annotation_file, index=False)
print(f"\n待标注文件已保存: {annotation_file}")
print(f"待标注数量: {len(annotation_export)} 条")

stats_df.to_csv(stats_file, encoding='utf-8-sig')
print(f"\n统计数据已保存: {stats_file}")

print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n下一步:")
print(f"1. 打开 '待标注评论.xlsx' 进行人工标注")
print(f"2. 标注字段说明:")
print(f"   - 情感: 积极/中性/消极")
print(f"   - 事件类型: 行情/业绩/造假/政策/产品/其他...")
print(f"3. 标注 {len(annotation_export)} 条后即可训练模型")