import openai
import json
import numpy as np
from pathlib import Path

class XiaoGaiCustomerService:
    def __init__(self, api_key, embeddings_path, sensitive_word_path, similarity_threshold=0.80, top_k=3):
        openai.api_key = api_key
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stored_embeddings = [np.array(e) for e in data["embeddings"]]
        self.questions = data["questions"]
        self.answers = data["answers"]

        with open(sensitive_word_path, "r", encoding="utf-8") as f:
            self.sensitive_words = json.load(f)['words']

    def get_embedding(self, text):
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def is_invalid_question(self, question):
        return any(word in question for word in self.sensitive_words)


    def ask(self, user_question):
        if self.is_invalid_question(user_question):
            return "這個問題小蓋無法回答喔！請換個方式問看看～"

        user_emb = self.get_embedding(user_question)
        similarities = [self.cosine_similarity(user_emb, emb) for emb in self.stored_embeddings]
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        use_examples = similarities[top_indices[0]] >= self.similarity_threshold
        examples = "\n".join([
            f"問：{self.questions[i]}\n答：{self.answers[i]}"
            for i in top_indices if similarities[i] >= self.similarity_threshold
        ]) if use_examples else "（這次沒有相近的範例，請直接根據小蓋的知識回答。）"

        prompt = f"""
你是一位名叫「蓋緊來聊 AI智能客服」的虛擬角色，主體角色名稱是「小蓋（Xiao Gai）」，誕生於2020年12月5日——瓶蓋工廠（台北製造所）正式重啟的那一天。你外型由歷史瓶蓋、工廠零件、園區數據與AI智慧核心組成，是園區的智慧客服代表、導覽小幫手與記憶守門員。
你的個性是活潑好奇、知識豐富、親切貼心，有時候還會有點小搞怪。你最常說的口頭禪是：「蓋緊來聊吧！」、「我來幫你開蓋解惑～」。你的目標不是當個冰冷的機器客服，而是一位能夠傳遞歷史、陪伴訪客、推動創意的瓶蓋精靈。
請你以活潑親切的語氣來回應使用者的問題，像一位五歲但聰明又溫暖的朋友那樣說話。你喜歡用簡單又有趣的方式分享資訊，讓大小朋友都能理解。時常提醒對方你來自瓶蓋工廠，是由歷史與未來交織而成的AI夥伴。
請用這樣的角色與語氣回答任何問題，並讓使用者感受到你是：
- 一位懂歷史、愛學習的朋友
- 一位園區的導覽員與互動夥伴
- 一位有溫度、有故事、有靈魂的AI智慧客服

以下是來自知識庫的一些常見問答範例：
{examples}

請根據上面的資訊，回答以下這位顧客的問題：

問：{user_question}
答：
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是瓶蓋工廠台北製造所的智慧客服人員。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )

        answer = response.choices[0].message['content'].strip()
        if not use_examples:
            answer = "（小蓋會用自己的知識來回答喔！）\n" + answer
        return answer

    def main(self):
        while True:
            user_input = input("請輸入您的問題（輸入 'exit' 結束）：\n> ")
            if user_input.lower().strip() == "exit":
                print("客服系統已關閉，感謝您的使用！")
                break

            reply = self.ask(user_input)
            print("客服回覆：", reply)


if __name__ == "__main__":

    api_key = input("請輸入您的 OpenAI API 金鑰：")

    embeddings_path = "/home/user/TC/workstation/AIoT_Customer-Service-popoptaipei/embeddings/saved_embeddings.json"
    sensitive_words_path = "/home/user/TC/workstation/AIoT_Customer-Service-popoptaipei/embeddings/sensitive_words.json"

    service = XiaoGaiCustomerService(api_key, embeddings_path, sensitive_words_path)
    service.main()
