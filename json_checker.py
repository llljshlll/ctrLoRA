import json

# JSON 파일 경로를 입력합니다.
json_file_path = "data/prompt.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    content = f.read()

print("파일 내용:")
print(content)

try:
    data = json.loads(content)
    print("JSON 데이터가 올바르게 로드되었습니다.")
except json.JSONDecodeError as e:
    print("JSON 디코딩 에러:", e)
