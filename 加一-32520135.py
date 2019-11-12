class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        
        if len(digits) == 0:
            return [1]
        
        digits.reverse()
        plus = False
        for i in range(len(digits)):
            digits[i] = digits[i] + 1
            if digits[i] > 9:
                digits[i] =  digits[i] - 10
                plus = True
            else:
                plus = False
                break
        
        if plus:
            digits.append(1)
        
        digits.reverse()
        return digits
            