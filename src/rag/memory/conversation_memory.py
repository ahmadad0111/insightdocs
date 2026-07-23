class ConversationMemory:

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def add(self, question, answer):

        self.history.append(question)

        self.history = self.history[-self.max_turns:]

    # def get_context(self):

    #     if not self.history:
    #         return ""

    #     text = ""

    #     for turn in self.history:

    #         text += (
    #             f"User: {turn['question']}\n"
    #             f"Assistant: {turn['answer']}\n\n"
    #         )

    #     return text

    def get_context(self):

        return "\n".join(self.history[-3:])