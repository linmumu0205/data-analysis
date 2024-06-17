import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv
import pandas as pd
import networkx as nx
from collections import Counter
import chardet
from matplotlib import font_manager
import re

# 检测文件编码
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 任务1：文本词频统计分析并可视化展示
def load_text(file_path):
    encoding = detect_file_encoding(file_path)
    print(f"Detected encoding: {encoding}")
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:  # 使用 errors='ignore' 忽略解码错误
        return file.read()

def preprocess_text(text):
    text = text.replace('\n', '').replace('\u3000', '')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 保留中文字符
    return text

def segment_text(text):
    return jieba.lcut(text)

def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        return set(file.read().split())

def filter_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords and len(word) > 1]

def calculate_word_frequency(words):
    return Counter(words)

def save_word_frequency(word_freq, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'frequency'])
        for word, freq in word_freq.items():
            writer.writerow([word, freq])

def chapter_statistics(text):
    chapters = text.split('第')[1:]
    stats = []
    for i, chapter in enumerate(chapters, 1):
        words = segment_text(chapter)
        stats.append({
            'chapter': i,
            'char_count': len(chapter),
            'word_count': len(words),
            'paragraph_count': chapter.count('\n')
        })
    return stats

def save_chapter_stats(stats, output_path):
    df = pd.DataFrame(stats)
    df.to_csv(output_path, index=False, encoding='utf-8')

def generate_wordcloud(word_freq, output_path):
    wc = WordCloud(
        width=800, height=600,
        background_color='white',
        font_path='simhei.ttf'  # 指定字体文件路径
    )
    wc.generate_from_frequencies(word_freq)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path)
    plt.show()

# 任务2：人物社交关系网络分析

def extract_characters(text, top_n=20):
    words = segment_text(text)
    stopwords = load_stopwords(stopwords_path)
    filtered_words = filter_stopwords(words, stopwords)
    word_freq = calculate_word_frequency(filtered_words)
    top_characters = [pair[0] for pair in word_freq.most_common(top_n)]
    return top_characters

def extract_characters_ner(text, top_n=20):
    import jieba.posseg as pseg
    words = pseg.cut(text)
    name_freq = Counter()
    for word, flag in words:
        if flag == 'nr':  # 'nr' 表示人名
            name_freq[word] += 1
    return [name for name, freq in name_freq.most_common(top_n)]

def build_relationships(text, characters):
    relationships = {char: Counter() for char in characters}
    paragraphs = text.split('\n')
    for para in paragraphs:
        for char in characters:
            if char in para:
                for other_char in characters:
                    if other_char in para and other_char != char:
                        relationships[char][other_char] += 1
    return relationships

def plot_social_network(relationships):
    G = nx.Graph()
    for char, related in relationships.items():
        for other_char, weight in related.items():
            G.add_edge(char, other_char, weight=weight)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))

    font_path = 'simhei.ttf'
    if font_path not in [f.fname for f in font_manager.fontManager.ttflist]:
        font_manager.fontManager.addfont(font_path)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=15, font_weight="bold",
            edge_color="#AAAAAA", font_family='SimHei', width=weights)
    plt.show()

def analyze_social_network(relationships):
    G = nx.Graph()
    for char, related in relationships.items():
        for other_char, weight in related.items():
            G.add_edge(char, other_char, weight=weight)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coefficient = nx.clustering(G)

    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)
    print("Clustering Coefficient:", clustering_coefficient)

def plot_character_relationships_over_time(text, characters, target_char):
    chapters = text.split('第')[1:]
    chapter_names = [f"第{i}章" for i in range(1, len(chapters) + 1)]
    char_relations = {char: [0] * len(chapters) for char in characters}

    for i, chapter in enumerate(chapters):
        for char in characters:
            if char != target_char and char in chapter and target_char in chapter:
                char_relations[char][i] += 1

    font_path = 'simhei.ttf'
    if font_path not in [f.fname for f in font_manager.fontManager.ttflist]:
        font_manager.fontManager.addfont(font_path)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for char, relations in char_relations.items():
        if char != target_char:
            plt.plot(chapter_names, relations, label=char)

    plt.xlabel('章节')
    plt.ylabel('关系权重')
    plt.title(f'{target_char}的关系网变化')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

def plot_character_frequency(text, characters):
    chapters = text.split('第')[1:]
    chapter_names = [f"第{i}章" for i in range(1, len(chapters) + 1)]
    char_freq = {char: [0] * len(chapters) for char in characters}

    for i, chapter in enumerate(chapters):
        for char in characters:
            char_freq[char][i] = chapter.count(char)

    font_path = 'simhei.ttf'
    if font_path not in [f.fname for f in font_manager.fontManager.ttflist]:
        font_manager.fontManager.addfont(font_path)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for char, freq in char_freq.items():
        plt.plot(chapter_names, freq, label=char)

    plt.xlabel('章节')
    plt.ylabel('出场频率')
    plt.title('主要人物出场频率')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

# 菜单选项

def menu():
    print("请选择任务:")
    print("1: 文本词频统计分析并可视化展示")
    print("2: 人物社交关系网络分析")
    print("3: 统计并可视化人物出场频率")
    print("4: 社交网络分析")
    print("5: 章节间的人物关系变化")
    choice = input("输入选项 (1/2/3/4/5): ")
    return choice

if __name__ == '__main__':
    # 路径设置
    text_path = '红楼梦.txt'
    stopwords_path = 'stopwords.txt'
    word_freq_path = 'word_frequency.csv'
    chapter_stats_path = 'chapter_stats.csv'
    wordcloud_path = 'wordcloud.png'

    # 加载文本
    text = load_text(text_path)
    processed_text = preprocess_text(text)

    choice = menu()

    if choice == '1':
        # 分词处理
        words = segment_text(processed_text)
        stopwords = load_stopwords(stopwords_path)
        filtered_words = filter_stopwords(words, stopwords)

        # 计算词频
        word_freq = calculate_word_frequency(filtered_words)
        save_word_frequency(word_freq, word_freq_path)

        # 章节统计
        stats = chapter_statistics(text)
        save_chapter_stats(stats, chapter_stats_path)

        # 词云展示
        generate_wordcloud(word_freq, wordcloud_path)

    elif choice == '2':
        # 提取人物名称
        characters = extract_characters(processed_text)

        # 构建社交关系
        relationships = build_relationships(processed_text, characters)
        plot_social_network(relationships)

    elif choice == '3':
        # 提取人物名称
        characters = extract_characters(processed_text)
        plot_character_frequency(processed_text, characters)

    elif choice == '4':
        # 提取人物名称
        characters = extract_characters(processed_text)

        # 构建社交关系
        relationships = build_relationships(processed_text, characters)
        analyze_social_network(relationships)

    elif choice == '5':
        # 提取人物名称
        characters = extract_characters(processed_text)
        target_char = input("输入目标人物名称: ")

        # 章节间的人物关系变化
        plot_character_relationships_over_time(processed_text, characters, target_char)
