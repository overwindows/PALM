class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = 2**len(nums)
        res = []
        for i in range(n):
            _res = []
            for j in range(len(nums)):
                if i % 2  == 1:
                    _res.append(nums[j])
                i = i // 2
            res.append(_res)
            
        return res