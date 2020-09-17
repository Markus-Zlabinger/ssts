from methods.method import Method
import editdistance

"""
Traditional Methods
"""


class LevenstheinMethod(Method):
    def similarity(self, sentence1, sentence2):
        sentence1 = self.preprocess(sentence1)
        sentence2 = self.preprocess(sentence2)
        d = editdistance.eval(sentence1, sentence2)
        longest = max(len(sentence1), len(sentence2))
        return 1 - d / longest


