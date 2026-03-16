"""
LLM 自动标注股吧评论
使用阿里云通义千问 tongyi-xiaomi-analysis-flash 模型
支持批量标注和 Few-shot 学习
"""

import os
import json
import time
import pandas as pd
import traceback
from tqdm import tqdm
from openai import OpenAI

# 配置日志文件
LOG_FILE = r"data\labeled_data\annotation.log"
ERROR_LOG_FILE = r"data\labeled_data\error.log"

def log_message(msg, level="INFO"):
    """记录日志"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level}] {msg}\n"
    print(log_line.strip())
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line)

def log_error(msg, exception=None):
    """记录错误日志"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [ERROR] {msg}\n"
    if exception:
        log_line += f"异常详情: {str(exception)}\n"
        log_line += f"堆栈跟踪:\n{traceback.format_exc()}\n"
    print(log_line.strip())
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line)

# ==================== 配置区 ====================

# API 配置 - 阿里云通义千问
API_KEY = "sk-9e5db6c446bb40f2b5c76fdd6dda6e08"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "tongyi-xiaomi-analysis-flash"  # 使用tongyi-xiaomi-analysis-flash模型

# 文件路径
LABELED_FILE = r"data\cleaned_data\标注评论.xlsx"
UNLABELED_FILE = r"data\cleaned_data\清洗后的全部评论.csv"
OUTPUT_DIR = r"data\labeled_data"
OUTPUT_FILE = r"data\labeled_data\LLM标注结果.csv"

# 断点续传配置
CHECKPOINT_FILE = r"data\labeled_data\checkpoint.json"  # 保存进度
AUTO_SAVE_INTERVAL = 5  # 每5批自动保存一次

# 标注数量（测试用100条，正式用None表示全部）
MAX_SAMPLES = None

# 是否标注全部数据（忽略已标注的Excel文件，从头开始标注全部）
ANNOTATE_ALL = True  # 设置为True标注全部81,262条，False则只标注未标注的部分

# 是否强制从头开始（删除已有检查点）
FORCE_RESTART = False  # 设置为True会删除已有检查点，从头开始标注

# Few-shot 示例数量
FEW_SHOT_SAMPLES = 3

# 批量标注数量（每次请求标注的评论数，控制token使用）
BATCH_SIZE = 5

# 请求间隔（秒）
REQUEST_DELAY = 1

# 最大重试次数
MAX_RETRIES = 3

# 重试等待时间（秒）
RETRY_WAIT = 2

# ==================== Prompt 设计 ====================

SYSTEM_PROMPT = """你是金融评论标注专家。请对每条评论进行标注。

标注维度：
1. 情感: 积极/中性/消极
2. 事件类型: 业绩相关/造假/欺诈/监管/处罚/重大事项/管理层/市场情绪/行业政策/其他
3. 置信度: 高/中/低 (标注的确定程度)
4. 关键词: 1-3个词

输出格式要求：
- 对每条评论输出一行JSON
- 格式: {"情感":"...","事件类型":"...","置信度":"...","关键词":"..."}
- 多条结果用换行分隔，不要输出其他内容"""


def build_few_shot_prompt(labeled_df):
    """构建Few-shot示例（精简版）"""
    examples = labeled_df.sample(n=min(FEW_SHOT_SAMPLES, len(labeled_df)), random_state=42)
    
    prompt = "示例:\n"
    
    for idx, row in examples.iterrows():
        text = row.get('清洗后评论', row.get('评论', row.get('content', '')))
        情感 = row.get('情感', row.get('sentiment', row.get('情感倾向', '')))
        事件类型 = row.get('事件类型', row.get('event_type', row.get('事件', '')))
        置信度 = row.get('置信度', row.get('confidence', row.get('确定程度', '')))
        关键词 = row.get('关键词', row.get('keyword', ''))
        
        prompt += f"评论: 「{str(text)[:50]}...」\n"
        prompt += f'{{"情感":"{情感}","事件类型":"{事件类型}","置信度":"{置信度}","关键词":"{关键词}"}}\n\n'
    
    prompt += "请标注以下评论（每行输出一个JSON）:\n"
    return prompt


def create_client():
    """创建LLM客户端"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def annotate_batch(client, texts, few_shot_prompt=""):
    """批量标注多条文本（控制token使用）- 带重试机制"""
    for attempt in range(MAX_RETRIES):
        try:
            # 构建批量标注的prompt
            user_content = few_shot_prompt
            for i, text in enumerate(texts, 1):
                user_content += f"{i}. 「{text}」\n"
            user_content += "\n请输出标注结果（每行一个JSON）："
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
                extra_body={"top_k": 1}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析多条JSON结果
            results = []
            for line in result_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 尝试直接解析
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError:
                    # 尝试提取JSON
                    start = line.find('{')
                    end = line.rfind('}') + 1
                    if start >= 0 and end > start:
                        try:
                            result = json.loads(line[start:end])
                            results.append(result)
                        except:
                            results.append(None)
                    else:
                        results.append(None)
            
            return results
                
        except Exception as e:
            error_msg = str(e)
            if "Connection error" in error_msg or "getaddrinfo failed" in error_msg:
                log_error(f"网络连接错误 (尝试 {attempt+1}/{MAX_RETRIES})", e)
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_WAIT * (attempt + 1)
                    log_message(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            else:
                log_error(f"批量标注错误", e)
            return [None] * len(texts)
    
    return [None] * len(texts)


def get_unlabeled_data():
    """获取未标注的数据"""
    print("读取已标注数据...")
    labeled_df = pd.read_excel(LABELED_FILE)
    print(f"已标注: {len(labeled_df)} 条")
    
    print("\n读取全部数据...")
    all_df = pd.read_csv(UNLABELED_FILE)
    print(f"总数据: {len(all_df)} 条")
    
    if ANNOTATE_ALL:
        # 标注全部数据
        print("\n⚠️  配置为标注全部数据（ANNOTATE_ALL=True）")
        print(f"将标注全部 {len(all_df)} 条评论")
        return labeled_df, all_df
    
    # 获取已标注的哈希值
    labeled_hashes = set()
    if '内容哈希' in labeled_df.columns:
        labeled_hashes = set(labeled_df['内容哈希'].dropna().tolist())
    elif '哈希' in labeled_df.columns:
        labeled_hashes = set(labeled_df['哈希'].dropna().tolist())
    
    # 过滤未标注的数据
    if labeled_hashes:
        unlabeled_df = all_df[~all_df['内容哈希'].isin(labeled_hashes)]
    else:
        # 如果没有哈希字段，取差集
        unlabeled_df = all_df.iloc[len(labeled_df):] if len(labeled_df) < len(all_df) else pd.DataFrame()
    
    print(f"未标注: {len(unlabeled_df)} 条")
    
    return labeled_df, unlabeled_df


def load_checkpoint():
    """加载检查点，恢复进度"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"✓ 发现检查点，已标注: {checkpoint.get('completed', 0)} 条")
            return checkpoint
        except:
            pass
    return {'completed': 0, 'results': [], 'failed': []}


def save_checkpoint(completed, results, failed):
    """保存检查点"""
    checkpoint = {
        'completed': completed,
        'results': results,
        'failed': failed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def batch_annotate(labeled_df, unlabeled_df):
    """批量标注 - 支持断点续传和随时启停"""
    client = create_client()
    
    # 限制标注数量
    if MAX_SAMPLES and len(unlabeled_df) > MAX_SAMPLES:
        unlabeled_df = unlabeled_df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"本次标注 {MAX_SAMPLES} 条")
    
    # 加载检查点
    checkpoint = load_checkpoint()
    start_batch = checkpoint.get('completed', 0) // BATCH_SIZE
    results = checkpoint.get('results', [])
    failed = checkpoint.get('failed', [])
    
    print(f"总共需要标注: {len(unlabeled_df)} 条")
    print(f"批量大小: {BATCH_SIZE} 条/次")
    print(f"从第 {start_batch + 1} 批开始标注")
    print(f"提示: 按 Ctrl+C 可随时停止，下次会自动续传\n")
    
    # 构建Few-shot提示词
    few_shot_prompt = build_few_shot_prompt(labeled_df)
    
    # 将数据分成批次
    total_batches = (len(unlabeled_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    batch_idx = start_batch
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    try:
        for batch_idx in tqdm(range(start_batch, total_batches), desc="批量标注进度", initial=start_batch, total=total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(unlabeled_df))
            batch_df = unlabeled_df.iloc[start_idx:end_idx]
            
            texts = batch_df['清洗后评论'].tolist()
            
            try:
                batch_results = annotate_batch(client, texts, few_shot_prompt)
                consecutive_errors = 0  # 重置错误计数
            except Exception as e:
                log_error(f"批次 {batch_idx+1} 标注失败", e)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    log_error(f"连续 {max_consecutive_errors} 次失败，停止标注")
                    break
                log_message(f"跳过批次 {batch_idx+1}，继续下一批...")
                failed.extend([{'index': idx, 'text': row['清洗后评论']} for idx, row in batch_df.iterrows()])
                time.sleep(RETRY_WAIT * 2)
                continue
            
            # 处理批次结果
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                result = batch_results[i] if i < len(batch_results) else None
                
                if result:
                    results.append({
                        'index': idx,
                        '股票名称': row['股票名称'],
                        '发布时间': row['发布时间'],
                        '清洗后评论': row['清洗后评论'],
                        '情感': result.get('情感', ''),
                        '事件类型': result.get('事件类型', ''),
                        '置信度': result.get('置信度', ''),
                        '关键词': result.get('关键词', '')
                    })
                else:
                    failed.append({'index': idx, 'text': row['清洗后评论']})
            
            # 控制请求速率
            time.sleep(REQUEST_DELAY)
            
            # 自动保存检查点
            if (batch_idx + 1) % AUTO_SAVE_INTERVAL == 0:
                completed = (batch_idx + 1) * BATCH_SIZE
                save_checkpoint(completed, results, failed)
                print(f"\n💾 检查点已保存: {completed} 条")
    
    except KeyboardInterrupt:
        log_message("用户中断标注", "WARNING")
        completed = batch_idx * BATCH_SIZE
        save_checkpoint(completed, results, failed)
        log_message(f"进度已保存到检查点，已标注: {completed} 条")
    except Exception as e:
        log_error(f"标注过程发生异常", e)
        completed = batch_idx * BATCH_SIZE if 'batch_idx' in locals() else 0
        save_checkpoint(completed, results, failed)
        log_message(f"异常时进度已保存，已标注: {completed} 条")
    
    log_message(f"标注完成! 成功: {len(results)}, 失败: {len(failed)}")
    
    # 保存失败的记录
    if failed:
        failed_file = os.path.join(OUTPUT_DIR, 'failed_annotation.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"失败的文本已保存到 {failed_file}")
    
    # 清除检查点（如果全部完成）
    if len(results) >= len(unlabeled_df) - len(failed):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("✓ 检查点已清除")
    
    return pd.DataFrame(results)


def show_statistics(df):
    """显示标注统计"""
    print("\n" + "=" * 50)
    print("标注统计")
    print("=" * 50)
    
    if len(df) == 0:
        print("没有成功标注的数据")
        return
    
    print(f"\n成功标注: {len(df)} 条")
    
    if '情感' in df.columns and df['情感'].notna().any():
        print("\n情感分布:")
        print(df['情感'].value_counts(normalize=True).round(3) * 100)
    
    if '事件类型' in df.columns and df['事件类型'].notna().any():
        print("\n事件类型分布:")
        print(df['事件类型'].value_counts())
    
    if '置信度' in df.columns and df['置信度'].notna().any():
        print("\n置信度分布:")
        print(df['置信度'].value_counts(normalize=True).round(3) * 100)
    
    if '股票名称' in df.columns and '情感' in df.columns:
        print("\n按股票统计:")
        print(pd.crosstab(df['股票名称'], df['情感']))


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 强制从头开始时删除检查点
    if FORCE_RESTART and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("⚠️  强制从头开始，已删除旧检查点")
    
    log_message("=" * 50)
    log_message(f"LLM 自动标注工具启动")
    log_message(f"模型: {MODEL}")
    log_message(f"API: {BASE_URL}")
    log_message("=" * 50)
    
    try:
        # 获取数据
        labeled_df, unlabeled_df = get_unlabeled_data()
        
        if len(unlabeled_df) == 0:
            log_message("没有需要标注的数据！", "WARNING")
            exit()
        
        # 批量标注
        result_df = batch_annotate(labeled_df, unlabeled_df)
    except Exception as e:
        log_error("主程序发生异常", e)
        raise
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if len(result_df) > 0:
        # 生成时间戳
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存主结果文件
        final_output = OUTPUT_FILE
        try:
            result_df.to_csv(final_output, index=False, encoding='utf-8-sig')
            print(f"\n✓ 结果已保存: {final_output}")
        except PermissionError:
            final_output = os.path.join(OUTPUT_DIR, f'LLM标注结果_{timestamp}.csv')
            result_df.to_csv(final_output, index=False, encoding='utf-8-sig')
            print(f"\n✓ 结果已保存: {final_output}")
        
        # 保存到归档文件夹
        archive_dir = os.path.join(OUTPUT_DIR, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        archive_file = os.path.join(archive_dir, f'LLM标注结果_{timestamp}.csv')
        result_df.to_csv(archive_file, index=False, encoding='utf-8-sig')
        print(f"✓ 归档已保存: {archive_file}")
        
        # 同时保存为Excel格式（方便查看）
        excel_file = os.path.join(archive_dir, f'LLM标注结果_{timestamp}.xlsx')
        try:
            result_df.to_excel(excel_file, index=False, engine='openpyxl')
            print(f"✓ Excel已保存: {excel_file}")
        except:
            print(f"⚠ Excel保存失败（可能需要安装openpyxl）")
        
        # 显示统计
        show_statistics(result_df)
    else:
        print("\n⚠ 没有成功标注的数据需要保存")
    
    print("\n" + "=" * 50)
    print("标注完成！")
    print("=" * 50)