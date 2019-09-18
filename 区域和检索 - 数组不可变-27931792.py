class NumArray:

    def __init__(self, nums: List[int]):
        self.cumSum = [0] * len(nums)
        self.nums = nums
        i = 0
        for num in nums:
            if i > 0:
                self.cumSum[i] = self.cumSum[i-1] + num
            else:
                self.cumSum[i] = num
            
            i=i+1
            
    def sumRange(self, i: int, j: int) -> int:
        return self.cumSum[j] - self.cumSum[i] + self.nums[i]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)