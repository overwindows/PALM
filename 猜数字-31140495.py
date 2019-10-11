class Solution:
    def game(self, guess: List[int], answer: List[int]) -> int:
        cnt = len(guess)
        ret = 0
        for i in range(cnt):
            if guess[i] == answer[i]:
                ret+=1
                
        return ret