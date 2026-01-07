import pandas as pd
import re
file_path = "/Users/shuqiwang/Downloads/我的阿勒泰小红书评论.xlsx"
df = pd.read_excel(file_path)
print(f"原始数据行数: {len(df)}")
df_cleaned = df.dropna(how='all').copy()
print(f"去除空白行后行数: {len(df_cleaned)}")
def clean_author(author_str):
    if pd.isna(author_str):
        return ""
    match = re.search(r'avatar/([^?]+)', str(author_str))
    if match:
        return match.group(1)
    return str(author_str)

df_cleaned['作者'] = df_cleaned['作者'].apply(clean_author)
def combine_title(title1, title2):
    if pd.isna(title1) and pd.isna(title2):
        return ""
    elif pd.isna(title1):
        return str(title2)
    elif pd.isna(title2):
        return str(title1)
    else:
        return f"{title1} | {title2}"

df_cleaned['完整标题'] = df_cleaned.apply(lambda row: combine_title(row['标题'], row['标题1']), axis=1)
def standardize_date(date_str):
    if pd.isna(date_str):
        return ""
    date_str = str(date_str).strip()
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    if re.match(r'\d{2}-\d{2}', date_str):
        return f"2024-{date_str}"
    if "天前" in date_str:
        return date_str  
    return date_str
df_cleaned['时间'] = df_cleaned['时间'].apply(standardize_date)
columns_to_keep = ['完整标题', '标题链接', 'cover_链接', '图片', '作者', '时间', 'count']
df_final = df_cleaned[columns_to_keep]
df_final = df_final.reset_index(drop=True)
output_path = "我的阿勒泰小红书评论_清洗后.xlsx"
df_final.to_excel(output_path, index=False)

print(f"数据清洗完成，保存至: {output_path}")
print(f"清洗后数据行数: {len(df_final)}")
