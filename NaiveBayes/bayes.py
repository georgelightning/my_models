import math
import re
import random
from typing import Tuple, Dict, Iterable, NamedTuple, List
from collections import defaultdict


# --- 1. Data Structure and Preprocessing ---

class Message(NamedTuple):
    text: str
    is_spam: bool


def tokenize(text: str) -> Dict[str, int]:

    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)

    word_counts = defaultdict(int)
    for word in all_words:
        word_counts[word] += 1

    return word_counts




def split_data(data: List[Message], test_ratio: float = 0.3) -> Tuple[List[Message], List[Message]]:
    random.seed(42)
    shuffled_data = data[:]
    random.shuffle(shuffled_data)

    test_size = int(len(shuffled_data) * test_ratio)

    test_set = shuffled_data[:test_size]
    train_set = shuffled_data[test_size:]

    return train_set, test_set


def evaluate_model(model: 'MultinomialNaiveBayes', test_set: List[Message], threshold: float = 0.5):
    true_positives = false_positives = true_negatives = false_negatives = 0

    for message in test_set:
        spam_prob = model.predict_spam_probability(message.text)
        predicted_spam = spam_prob >= threshold

        if predicted_spam:
            if message.is_spam:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if message.is_spam:
                false_negatives += 1
            else:
                true_negatives += 1

    total = len(test_set)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0

    precision_denominator = true_positives + false_positives
    precision = true_positives / precision_denominator if precision_denominator > 0 else 0

    recall_denominator = true_positives + false_negatives
    recall = true_positives / recall_denominator if recall_denominator > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
    }


def classify_message(model: 'MultinomialNaiveBayes', text: str, threshold: float = 0.5) -> str:

    spam_prob = model.predict_spam_probability(text)

    if spam_prob >= threshold:
        classification = "SPAM"
    else:
        classification = "HAM"

    return f"Spam Probability = {spam_prob:.4f}, Classification: {classification}"



class MultinomialNaiveBayes:

    def __init__(self, k: float = 1.0) -> None:
        self.k = k
        self.tokens: set = set()

        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)

        self.spam_messages = self.ham_messages = 0
        self.total_spam_words = 0
        self.total_ham_words = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in re.findall("[a-z0-9]+", message.text.lower()):
                self.tokens.add(token)

                if message.is_spam:
                    self.token_spam_counts[token] += 1
                    self.total_spam_words += 1
                else:
                    self.token_ham_counts[token] += 1
                    self.total_ham_words += 1

    def _likelihoods(self, token: str) -> Tuple[float, float]:
        v_size = len(self.tokens)

        spam_denominator = self.total_spam_words + self.k * v_size
        ham_denominator = self.total_ham_words + self.k * v_size

        spam_lik = (self.token_spam_counts[token] + self.k) / spam_denominator
        ham_lik = (self.token_ham_counts[token] + self.k) / ham_denominator

        return spam_lik, ham_lik

    def predict_spam_probability(self, text: str) -> float:
        text_tokens = tokenize(text)

        num_messages = self.spam_messages + self.ham_messages

        if num_messages == 0:
            return 0.5

        p_spam = self.spam_messages / num_messages
        p_ham = self.ham_messages / num_messages

        log_prob_if_spam = math.log(p_spam)
        log_prob_if_ham = math.log(p_ham)

        for token, count in text_tokens.items():
            if token in self.tokens:
                spam_lik, ham_lik = self._likelihoods(token)

                log_prob_if_spam += count * math.log(spam_lik)
                log_prob_if_ham += count * math.log(ham_lik)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham)



BASE_SPAM = [

    "FREE money guaranteed, click here NOW! Review this offer.",
    "Congratulations, you won a FREE vacation! Claim your prize now. Attached is a plan.",
    "URGENT: Your account is suspended. Click immediately to restore access. Please review.",
    "100% guarantee to earn easy CASH from home. Limited time offer. Check the schedule.",
    "WINNER! Click on this link to get your million dollar cash prize. See the report.",
    "Attention: Your Netflix payment failed. Update details quickly or account disabled. Call today.",
    "Huge savings on loans. Apply today and save money. Review this.",
    "Investment opportunity: High returns guaranteed, secret information inside. Final report.",
    "We need to confirm your identity. Click here to avoid termination. Review the details.",
    "Get rich quick scheme! Learn the secret to unlimited wealth today. Schedule a meeting.",
]

BASE_HAM = [
    "Meeting tomorrow at 10 AM, please review the financial report. Can you make it?",
    "Please review the attached project plan and send feedback by Friday. It's urgent.",
    "The team needs to confirm the budget plan for next week's meeting. Get a free quote.",
    "Did you receive the updated version of the budget report? Let me know. No need to click.",
    "Thank you for your assistance with the report. I will call you tomorrow. This is urgent.",
    "Please remember to approve the time-off requests before the end of the day. Free to call.",
    "The updated project schedule is attached. Review before our next discussion. Limited offer.",
    "I received your email regarding the HR policy change. I will review it. Urgent response needed.",
    "Let's finalize the presentation slides for the client meeting on Tuesday. Free to update.",
    "The server maintenance window is scheduled for Saturday at midnight. Get cash back.",
]


def generate_full_data(n_per_class=100) -> List[Message]:
    """Generates 2*n_per_class messages by sampling and varying base phrases."""
    full_list = []

    # Generate SPAM (100 messages)
    for i in range(n_per_class):
        base_text = random.choice(BASE_SPAM)
        variation = f" {random.choice(['Act now', 'Immediate action required', 'Offer ends soon'])}!" if i % 5 == 0 else ""
        full_list.append(Message(f"{base_text}{variation}", is_spam=True))

    # Generate HAM (100 messages)
    for i in range(n_per_class):
        base_text = random.choice(BASE_HAM)
        variation = f" {random.choice(['Can you confirm?', 'Is this possible?', 'Follow up needed'])}?" if i % 5 == 0 else ""
        full_list.append(Message(f"{base_text}{variation}", is_spam=False))

    return full_list



train_data = generate_full_data(n_per_class=100)


full_data = train_data
train_set, test_set = split_data(full_data, test_ratio=0.3)


model = MultinomialNaiveBayes(k=1.0)
model.train(train_set)


metrics = evaluate_model(model, test_set, threshold=0.5)



print(f"--- Model Validation Results ---")
print(f"Total Training Messages: {len(train_set)}")
print(f"Total Test Messages: {len(test_set)}")
print("-" * 30)

print(f"Accuracy: {metrics['Accuracy']:.4f}")
print(f"Precision (Spam): {metrics['Precision']:.4f}")
print(f"Recall (Spam): {metrics['Recall']:.4f}")
print("-" * 30)
print(f"True Positives (Spam Correctly Caught): {metrics['TP']}")
print(f"False Positives (Ham incorrectly Labeled Spam): {metrics['FP']}")


new_message = "Please review the urgent financial offer and click the attached report."
print("\n--- Example Prediction (Gray Area) ---")
print(f"Message: '{new_message}'")
print(f"Result: {classify_message(model, new_message)}")

