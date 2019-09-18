class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        cnt = {}
        for num in nums:
            if num in cnt:
                cnt[num] += 1
            else:
                cnt[num] = 1
        
        for k,v in cnt.items():
            if v == 1:
                return k
        