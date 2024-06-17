import chardet

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 检查文件编码
text_path = '红楼梦.txt'
encoding = detect_file_encoding(text_path)
print(f"文件编码格式: {encoding}")
