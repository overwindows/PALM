class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        size = len(nums)
        pivot = 0
        ptr = 1
        cnt = 0
        while ptr < size and pivot < ptr:
            #print(pivot,ptr)
            while ptr<size and nums[pivot] == nums[ptr] :
                cnt+=1
                ptr+=1
            if ptr< size:
                pivot += 1
                nums[pivot] = nums[ptr]
                ptr += 1

        for _ in range(cnt):
            nums.pop()
        return None 
        