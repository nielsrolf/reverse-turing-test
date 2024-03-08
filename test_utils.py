import json
from utils import fix_json


def test_fix_json():
    import random
    # build a random nested json structure
    data = ['a', 'b', 'c']
    for _ in range(3):
        data = {i: j for i, j in enumerate(data)}
        data = [data] * 3
    data = json.dumps(data)
    
    # break it and parse it
    for _ in range(10):
        broken = data[:random.randint(0, len(data))]
        parsed = fix_json(broken)
        print(json.dumps(parsed))


if __name__ == "__main__":
    test_fix_json()