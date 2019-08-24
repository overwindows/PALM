class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        if len(nums) == 1:
            return [nums]
        for x in nums:
            xx = [x]
            # print(xx)
            _nums = nums.copy()
            _nums.remove(x)
            ll = self.permute(_nums)
            for l in ll:
                res.append(xx+l)
        return res