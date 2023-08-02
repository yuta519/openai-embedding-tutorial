import json

from services.openai_service import OpenAiService

demo_data = ["アイス", "うどん", "カレーライス", "焼肉", "そうめん"]


def main():
    service = OpenAiService()
    index = read_json_file()
    search_result = [
        {
            "title": i["title"],
            "similarity": service.cosine_similarity(
                service.createVector("冷たい食べ物"), i["embedding"]
            ),
        }
        for i in index
    ]
    print(search_result)


def write_to_file(index):
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def read_json_file():
    with open("index.json") as f:
        index = json.load(f)

    return index


if __name__ == "__main__":
    main()
