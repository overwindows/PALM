class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        len_n = len(nums)
        st = 0
        ed = len_n-1
        if len_n == 0:
            return [-1,-1]
        while st < ed:
            mid = (st+ed)//2
            if target >= nums[mid]:
                st = mid+1
                if target == nums[mid] and nums[st] != target:
                    st = mid
                    break
            elif target < nums[mid]:
                ed = mid-1
        _ed = st
        
        st = 0
        ed = len_n-1
        while st < ed:
            mid = (st+ed)//2
            if target > nums[mid]:
                st = mid+1
            elif target <= nums[mid]:
                ed = mid-1
                if mid == 0:
                    ed = mid
                    break
                if target == nums[mid] and target > nums[ed]:
                    ed = mid
                    break
        _st = ed
        
        #print(_st,_ed)
        if nums[_st] == target:
            return(_st,_ed)
        else:
            return [-1,-1]
        