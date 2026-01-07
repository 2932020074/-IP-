import pandas as pd
import numpy as np
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
df = pd.read_excel('/Users/shuqiwang/Downloads/我的阿勒泰小红书评论.xlsx')
print("数据形状:", df.shape)
print("\n前几行数据:")
print(df.head())
print("\n列名:")
print(df.columns.tolist())
print("\n数据基本信息:")
print(df.info())
df_cleaned = df.dropna(subset=['标题', '标题1']).reset_index(drop=True)
def analyze_sentiment(text):
    """对文本进行情感分析"""
    if pd.isna(text) or str(text).strip() == '':
        return None, None
    try:
        s = SnowNLP(str(text))
        score = s.sentiments
        if score > 0.6:
            sentiment = '积极'
        elif score < 0.4:
            sentiment = '消极'
        else:
            sentiment = '中性'
        return score, sentiment
    except Exception as e:
        print(f"情感分析错误: {e}, 文本: {text[:50]}...")
        return None, None
print("\n正在对笔记内容进行情感分析...")
sentiment_results = []
for text in df_cleaned['标题1']:
    score, sentiment = analyze_sentiment(text)
    sentiment_results.append({
        'score': score,
        'sentiment': sentiment
    })
sentiment_df = pd.DataFrame(sentiment_results)
df_cleaned['情感得分'] = sentiment_df['score']
df_cleaned['情感分类'] = sentiment_df['sentiment']
print("\n=== 情感分析结果 ===")
sentiment_counts = df_cleaned['情感分类'].value_counts()
print(f"情感分布:\n{sentiment_counts}")
print(f"\n积极率: {sentiment_counts.get('积极', 0) / len(df_cleaned) * 100:.1f}%")
print(f"平均情感得分: {df_cleaned['情感得分'].mean():.3f}")
print("\n=== 按作者情感分析 ===")
author_sentiment = df_cleaned.groupby('作者').agg({
    '情感得分': ['mean', 'count'],
    'count': 'sum'
}).round(3)
author_sentiment.columns = ['平均情感分', '发帖数', '总互动数']
author_sentiment = author_sentiment.sort_values('平均情感分', ascending=False)

print(f"最积极的10位作者:")
print(author_sentiment.head(10))
print(f"\n最消极的10位作者:")
print(author_sentiment.tail(10))
print("\n=== 按时间情感趋势 ===")
def parse_time(time_str):
    try:
        if pd.isna(time_str):
            return None
        if isinstance(time_str, str):
            if '天前' in time_str:
                days = int(time_str.replace('天前', ''))
                return datetime.now() - pd.Timedelta(days=days)
            elif '月' in time_str and '日' not in time_str:
                if '-' in time_str and len(time_str) <= 5:
                    month, day = time_str.split('-')
                    return datetime(2024, int(month), int(day))
            else:
                return pd.to_datetime(time_str, errors='coerce')
        return time_str
    except:
        return None

df_cleaned['发布时间'] = df_cleaned['时间'].apply(parse_time)
if df_cleaned['发布时间'].notna().any():
    df_cleaned['月份'] = df_cleaned['发布时间'].dt.to_period('M')
    monthly_sentiment = df_cleaned.groupby('月份').agg({
        '情感得分': 'mean',
        '标题': 'count'
    }).round(3)
    
    monthly_sentiment.columns = ['平均情感分', '发帖数']
    print("月度情感趋势:")
    print(monthly_sentiment)
print("\n=== 高互动内容分析 ===")
df_cleaned['互动数'] = pd.to_numeric(df_cleaned['count'], errors='coerce')

top_interaction = df_cleaned.nlargest(10, '互动数')[['标题1', '作者', '互动数', '情感得分', '情感分类']]
print("互动最高的10条笔记:")
print(top_interaction[['标题1', '互动数', '情感分类']])
print("\n=== 情感与互动关系 ===")
sentiment_interaction = df_cleaned.groupby('情感分类').agg({
    '互动数': ['mean', 'sum', 'count']
}).round(2)

sentiment_interaction.columns = ['平均互动数', '总互动数', '帖子数']
print("不同情感的互动情况:")
print(sentiment_interaction)
output_file = '我的阿勒泰小红书评论_情感分析.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_cleaned.to_excel(writer, sheet_name='情感分析结果', index=False)
    author_sentiment.to_excel(writer, sheet_name='作者情感统计')
    if 'monthly_sentiment' in locals():
        monthly_sentiment.to_excel(writer, sheet_name='月度趋势')
    sentiment_interaction.to_excel(writer, sheet_name='情感互动分析')

print(f"\n分析完成! 结果已保存到: {output_file}")
print("\n生成可视化图表...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
colors = {'积极': '#4CAF50', '中性': '#FFC107', '消极': '#F44336'}
sentiment_counts = df_cleaned['情感分类'].value_counts()
axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
              colors=[colors.get(x, '#999999') for x in sentiment_counts.index])
axes[0, 0].set_title('情感分类分布', fontsize=14, fontweight='bold')
axes[0, 1].hist(df_cleaned['情感得分'].dropna(), bins=20, color='#2196F3', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('情感得分', fontsize=12)
axes[0, 1].set_ylabel('频次', fontsize=12)
axes[0, 1].set_title('情感得分分布直方图', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].scatter(df_cleaned['情感得分'], df_cleaned['互动数'], 
                  alpha=0.6, c=df_cleaned['情感得分'], cmap='RdYlGn')
axes[1, 0].set_xlabel('情感得分', fontsize=12)
axes[1, 0].set_ylabel('互动数', fontsize=12)
axes[1, 0].set_title('情感得分 vs 互动数', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
if not sentiment_interaction.empty:
    bars = axes[1, 1].bar(range(len(sentiment_interaction)), 
                         sentiment_interaction['平均互动数'].values,
                         color=[colors.get(x, '#999999') for x in sentiment_interaction.index])
    axes[1, 1].set_xticks(range(len(sentiment_interaction)))
    axes[1, 1].set_xticklabels(sentiment_interaction.index)
    axes[1, 1].set_ylabel('平均互动数', fontsize=12)
    axes[1, 1].set_title('不同情感的平均互动数', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, sentiment_interaction['平均互动数'].values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('情感分析可视化.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n" + "="*50)
print("《我的阿勒泰》小红书评论情感分析报告")
print("="*50)

print(f"""
数据分析概览：
1. 数据总量：{len(df_cleaned)}条有效笔记评论
2. 总体情感倾向：{'积极' if df_cleaned['情感得分'].mean() > 0.5 else '中性' if df_cleaned['情感得分'].mean() > 0.4 else '消极'}
3. 平均情感得分：{df_cleaned['情感得分'].mean():.3f}
4. 情感分布：
   - 积极评论：{sentiment_counts.get('积极', 0)}条 ({sentiment_counts.get('积极', 0)/len(df_cleaned)*100:.1f}%)
   - 中性评论：{sentiment_counts.get('中性', 0)}条 ({sentiment_counts.get('中性', 0)/len(df_cleaned)*100:.1f}%)
   - 消极评论：{sentiment_counts.get('消极', 0)}条 ({sentiment_counts.get('消极', 0)/len(df_cleaned)*100:.1f}%)

关键发现：
1. 最积极的作者：{author_sentiment.index[0] if len(author_sentiment) > 0 else 'N/A'}（平均情感分：{author_sentiment['平均情感分'].iloc[0] if len(author_sentiment) > 0 else 'N/A'}）
2. 最高互动笔记：{top_interaction.iloc[0]['标题1'][:50] if len(top_interaction) > 0 else 'N/A'}...
   互动数：{top_interaction.iloc[0]['互动数'] if len(top_interaction) > 0 else 'N/A'}，情感：{top_interaction.iloc[0]['情感分类'] if len(top_interaction) > 0 else 'N/A'}
3. 情感与互动关系：{'积极' if sentiment_interaction.loc['积极', '平均互动数'] > sentiment_interaction.loc['消极', '平均互动数'] else '消极'}情感的内容获得更高平均互动

建议：
1. 重点关注积极情感内容的特点和表达方式
2. 分析高互动内容的情感特征和话题倾向
3. 关注消极情感反馈，了解用户不满意的方面
""")
print("\n=== 文本情感分析示例 ===")
sample_comments = df_cleaned.sample(min(5, len(df_cleaned)), random_state=42)
for idx, row in sample_comments.iterrows():
    print(f"\n示例 {idx+1}:")
    print(f"标题: {row['标题1'][:60]}...")
    print(f"作者: {row['作者']}")
    print(f"情感得分: {row['情感得分']:.3f} ({row['情感分类']})")
    print(f"互动数: {row['互动数']}")
