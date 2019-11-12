class Solution:
    def titleToNumber(self, s: str) -> int:
        offset = ord('A')-1
        nums = list(s)
        nums.reverse()
        
        outputs = 0
        
        #321
        #1110001
        
        lens = len(nums)
        print(nums)
        for i in range(lens):
            #print(outputs,(ord(nums[i]) - offset),)
            outputs += (ord(nums[i]) - offset) * (26**i)
            #print(outputs)
            
        return outputs
        
        