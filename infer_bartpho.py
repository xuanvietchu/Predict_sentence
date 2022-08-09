from transformers import MBartTokenizerFast, MBartForConditionalGeneration, AutoTokenizer
import torch
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("/home/vietchu/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 8
model_path = f"./models/bartpho/epoch_{epoch_num}"

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)

DISTANCE_THRESHOLD = 1
def main(sent, num_outputs: int = 1, max_length: int = 200):
    generated = model.generate(
        tokenizer.encode(sent, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt').to(device),
        num_beams=20, num_return_sequences=num_outputs, max_length=max_length
    )

    result = []
    for generated_sentence in generated:
        out = tokenizer.decode(
                generated_sentence,
                skip_special_tokens=True
            )
        result.append(out)
        print(out + "\n")

    return result

def levenshteinDistance(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    final_distance = distances[-1]
    score = final_distance / max(len(s1), len(s2))
    return {
        "raw_distance": final_distance,
        "score": score
    }

def filter_answer(raw_sentence, paraphrase_sentences):
    output = []
    for sen in paraphrase_sentences:
        distance_score = levenshteinDistance(raw_sentence, sen)
        raw_distance = distance_score["raw_distance"]
        score = distance_score["score"]
        if raw_distance >= 0.02*len(raw_sentence):
            output.append((raw_distance, sen))
    output.sort(reverse=True)
    
    return output

def tokenize_input(raw_sentence):
    endline = ['.', '?', '!']
    if raw_sentence[-1] not in endline:
        raw_sentence += '.'

    text = rdrsegmenter.tokenize(raw_sentence)
    for i, tex in enumerate(text): 
        text[i] = ' '.join(tex)
    text = ' '.join(text)

    return text

if __name__ == "__main__":
    text = "Đĩa đơn No one at all được phát hành ngày 15/4 trên Spotify và MV được phát hành ngày 16/4 trên Youtube"
    main(text)