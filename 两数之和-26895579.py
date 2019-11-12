class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        _nums = nums[:]
        nums.sort()
        st = 0
        ed = len(nums)-1
        
        while st<ed:
          two_sum = nums[st] + nums[ed]
          if two_sum == target:
            break
          
          if two_sum > target:
            ed -= 1
          else:
            st += 1
        
        val0 = nums[st]
        val1 = nums[ed]
        
        ix0 = _nums.index(val0)
        if val0 == val1:
          ix1 = _nums.index(val0,ix0+1)
        else:
          ix1 = _nums.index(val1)
        
        return [ix0,ix1]