"""
pred.py

This script cleans the input test CSV file using the same steps as used during training,
loads pre-saved decision tree CSV files (one per tree), and uses them in a random forest
ensemble (via majority voting) to generate predictions.

# Some logic or suggestions in this script were generated with the assistance of ChatGPT (OpenAI, 2025).
# https://openai.com/chatgpt

Usage:
    python pred.py <test_csv_filename>
"""

import sys
import re
from collections import Counter

import numpy as np
import pandas as pd

# ============================
# Data Cleaning Function
# ============================
def clean_all_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise e

    # --- Feature 1: Clean Q1 ---
    def clean_q1(text):
        if pd.isnull(text):
            return None
        text = str(text).strip().lower()
        match = re.search(r'\b([1-5])\b', text)
        if match:
            return int(match.group(1))
        return None

    q1_col = None
    for col in df.columns:
        if 'q1' in col.strip().lower():
            q1_col = col
            break
    if q1_col is None:
        raise ValueError("Q1 column not found.")
    df[q1_col] = df[q1_col].apply(clean_q1)
    df[q1_col] = df[q1_col].fillna(df[q1_col].median())

    # --- Feature 2: Clean Q2 ---
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12
    }
    def words_to_digits(text):
        for word, digit in word_to_number.items():
            pattern = r'\b' + word + r'\b'
            text = re.sub(pattern, str(digit), text)
        return text

    def clean_q2(text):
        if pd.isnull(text):
            return None
        text = str(text).lower().strip()
        text = words_to_digits(text)
        text = re.sub(r'[:;]', '', text)
        text = re.sub(r'[$€£]|cad|canadian|dollars?|usd', '', text)
        text = re.sub(r'\s+', ' ', text)
        range_pattern = r'(\d+(?:\.\d+)?)\s*(?:-|–|to|or|~|—)\s*(\d+(?:\.\d+)?)'
        range_match = re.search(range_pattern, text)
        if range_match:
            try:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                return round((low + high) / 2)
            except:
                pass
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        ing_match = re.search(r'(\d+(?:\.\d+)?)\s+ingredients?', text)
        if ing_match:
            return round(float(ing_match.group(1)))
        if ',' in text or ' and ' in text:
            items = re.split(r',| and ', text)
            items = [item.strip() for item in items if item.strip()]
            if items:
                return len(items)
        if len(numbers) > 1:
            return round(sum(map(float, numbers)) / len(numbers))
        elif len(numbers) == 1:
            return round(float(numbers[0]))
        vague_words = ['many', 'a lot', 'several', 'tons', 'bunch', 'plenty', 'idk', 'don’t know', "don't know", 'not sure']
        if any(word in text for word in vague_words):
            return None
        return None

    q2_col = next((col for col in df.columns if 'q2' in col.strip().lower()), None)
    if q2_col is None:
        raise ValueError("Q2 column not found.")
    df[q2_col] = df[q2_col].apply(clean_q2)
    df[q2_col] = df[q2_col].fillna(df[q2_col].median())

    # --- Feature 3: Clean Q3 ---
    setting_options = {
        'is_Week_day_lunch': 'week day lunch',
        'is_Week_day_dinner': 'week day dinner',
        'is_Weekend_lunch': 'weekend lunch',
        'is_Weekend_dinner': 'weekend dinner',
        'is_At_a_party': 'at a party',
        'is_Late_night_snack': 'late night snack'
    }
    q3_col = None
    for col in df.columns:
        if 'q3' in col.strip().lower():
            q3_col = col
            break
    if q3_col is None:
        raise ValueError("Q3 column not found.")
    for new_col, keyword in setting_options.items():
        df[new_col] = df[q3_col].apply(lambda x: 1 if isinstance(x, str) and keyword in x.lower() else 0)
    df.drop(columns=[q3_col], inplace=True)

    # --- Feature 4: Clean Q4 ---
    def clean_q4(text):
        if pd.isnull(text):
            return None
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[$€£]|cad|canadian|dollars?|usd', '', text)
        text = text.replace('~', '-')
        range_pattern = r'(\d+(?:\.\d+)?)\s*(?:-|to|or|–|—)\s*(\d+(?:\.\d+)?)'
        range_match = re.search(range_pattern, text)
        if range_match:
            try:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                return round((low + high) / 2)
            except:
                pass
        all_numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if all_numbers:
            all_numbers = [float(num) for num in all_numbers]
            avg = sum(all_numbers) / len(all_numbers)
            return round(avg)
        return None

    q4_col = next((col for col in df.columns if 'q4' in col.strip().lower()), None)
    if q4_col is None:
        raise ValueError("Q4 column not found.")
    df[q4_col] = df[q4_col].apply(clean_q4)
    df[q4_col] = df[q4_col].fillna(df[q4_col].median())

    # --- Feature 5: Process Q5 into Bag-of-Words ---
    from collections import Counter
    def tokenize(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    def process_q5_bow(df, top_n=350, binary=True):
        q5_vocab = np.load("q5_vocab.npy", allow_pickle=True).tolist()
        q5_col = next((col for col in df.columns if 'q5' in col.strip().lower()), None)
        if q5_col is None:
            raise ValueError("Q5 column not found.")
        token_lists = df[q5_col].fillna('').apply(tokenize)

        bow_features = []
        for tokens in token_lists:
            token_counts = Counter(tokens)
            row = {f"q5_bow_{word}": int(word in token_counts) if binary else token_counts[word] for word in q5_vocab}
            bow_features.append(row)

        bow_df = pd.DataFrame(bow_features)
        df = df.drop(columns=[q5_col]).reset_index(drop=True)
        df = pd.concat([df, bow_df], axis=1)
        return df
    df = process_q5_bow(df, top_n=350, binary=True)

    # --- Feature 6: Clean Q6 (Drink Categories) ---
    q6_col = next((col for col in df.columns if 'q6' in col.strip().lower()), None)
    if q6_col is None:
        raise ValueError("Q6 column not found.")
    drink_keywords = {
        'is_soda': ['soda', 'cola', 'coke', 'coca', 'sprite', 'fanta', 'root beer', 'dr pepper', 'pepsi', '7up', 'mountain dew', 'cream soda', 'seven up'],
        'is_juice': ['juice', 'orange juice', 'apple juice', 'grape juice', 'cranberry', 'lemonade'],
        'is_energy': ['red bull', 'monster', 'rockstar', 'energy drink'],
        'is_water': ['water', 'bottled water', 'tap water', 'sparkling water'],
        'is_tea': ['tea', 'iced tea', 'green tea', 'bubble tea'],
        'is_coffee': ['coffee', 'latte', 'espresso', 'cappuccino', 'cold brew'],
        'is_milk': ['milk', 'chocolate milk', 'almond milk', 'soy milk', 'oat milk']
    }
    def detect_drink_category(text, keywords):
        if not isinstance(text, str):
            return 0
        text = text.lower()
        return 1 if any(kw in text for kw in keywords) else 0
    for col_name, keywords in drink_keywords.items():
        df[col_name] = df[q6_col].apply(lambda x: detect_drink_category(x, keywords))
    df.drop(columns=[q6_col], inplace=True)

    # --- Feature 7: Clean Q7 (Group Categories) ---
    q7_col = None
    for col in df.columns:
        if 'q7' in col.strip().lower():
            q7_col = col
            break
    if q7_col is None:
        raise ValueError("Q7 column not found.")
    q7_targets = {
        'is_Parents': 'parents',
        'is_Siblings': 'siblings',
        'is_Friends': 'friends',
        'is_Teachers': 'teachers',
        'is_Strangers': 'strangers'
    }
    for new_col, keyword in q7_targets.items():
        df[new_col] = df[q7_col].apply(lambda x: 1 if isinstance(x, str) and keyword in x.lower() else 0)
    df.drop(columns=[q7_col], inplace=True)

    # --- Feature 8: Clean Q8 (Spice Level) ---
    q8_col = None
    for col in df.columns:
        if 'q8' in col.strip().lower():
            q8_col = col
            break
    if q8_col is None:
        raise ValueError("Q8 column not found.")
    def map_spice_level(text):
        if not isinstance(text, str):
            return 0
        t = text.lower()
        if 'none' in t:
            return 0
        elif 'mild' in t or 'little' in t:
            return 1
        elif 'moderate' in t or 'medium' in t:
            return 2
        elif 'hot' in t and 'lot' in t:
            return 3
        elif 'my hot sauce' in t:
            return 4
        else:
            return 0
    df['spice_level'] = df[q8_col].apply(map_spice_level)
    df.drop(columns=[q8_col], inplace=True)

    # Drop the ID column if present
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    return df

# ============================
# Load Pre-saved Artifacts
# ============================
expected_columns = np.load("feature_order.npy", allow_pickle=True).tolist()
manual_forest_raw = np.load("manual_forest.npy", allow_pickle=True)
manual_forest = [
    tree for tree in manual_forest_raw
    if isinstance(tree, dict) and len(tree) > 0
]

# ============================
# Prediction Functions
# ============================
def predict_tree(x, tree_dict, node_id=0):
    node = tree_dict[node_id]
    if node['left_child'] == -1 and node['right_child'] == -1:
        return int(np.argmax(node['value'][0]))

    feature_index = int(node['feature'])
    threshold = float(node['threshold'])

    try:
        value = float(x[feature_index])
    except:
        raise ValueError(f"Invalid feature value at index {feature_index}: {x[feature_index]}")

    if value <= threshold:
        return predict_tree(x, tree_dict, int(node['left_child']))
    else:
        return predict_tree(x, tree_dict, int(node['right_child']))

def predict_random_forest(x, trees_list):
    votes = [predict_tree(x, tree) for tree in trees_list]
    return Counter(votes).most_common(1)[0][0]

# ============================
# Main Prediction Wrapper
# ============================
def predict_all(test_file):
    df = clean_all_data(test_file)

    # Drop label for prediction (but keep a copy for eval)
    label_col = df["Label"] if "Label" in df.columns else None
    if label_col is not None:
        df = df.drop(columns=["Label"])

    # Align feature columns
    df = df.reindex(columns=expected_columns, fill_value=0)
    X = df.values

    # Validate compatibility
    valid_feature_indices = []
    for tree in manual_forest:
        features = [int(node['feature']) for node in tree.values()
                    if isinstance(node, dict) and node.get('feature', -2) != -2]
        if features:
            valid_feature_indices.append(max(features))

    if not valid_feature_indices:
        raise ValueError("No valid feature indices found in forest.")
    max_feature_index = max(valid_feature_indices)

    if X.shape[1] <= max_feature_index:
        raise ValueError(f"Expected at least {max_feature_index + 1} features, got {X.shape[1]}.")

    predictions = [predict_random_forest(x, manual_forest) for x in X]
    label_mapping = {
    0: "Pizza",
    1: "Shawarma",
    2: "Sushi"
    }
    pred_labels = [label_mapping[p] for p in predictions]
    
    return pred_labels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pred.py <test_csv_filename>")
        sys.exit(1)

    test_file = sys.argv[1]
    predictions = predict_all(test_file)

    # ✅ Print predictions
    for p in predictions:
        print(p)

    # # ✅ If labels present, compute accuracy
    # if true_labels is not None:
    #     label_to_num = {"Pizza": 0, "Shawarma": 1, "Sushi": 2}
    #     y_true = [label_to_num.get(label, -1) for label in true_labels]
    #     y_pred = [label_to_num[p] for p in predictions]
    #     acc = np.mean(np.array(y_pred) == np.array(y_true))
    #     print(f"\n✅ Accuracy on provided data: {acc:.4f}")