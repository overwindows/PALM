class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 1,2,3,4,5 -> 1,2,3,5,4
        # 2,1,5,4,3 -> 2,3,1,4,5
        
        lens = len(nums)
        if lens < 2:
            return nums
        
        flag = False
        for i in range(1,lens):
            if flag:
                break
            partial = False
            for j in range(i+1,lens+1):
                if nums[lens-i] > nums[lens-j]:
                    if not partial:
                        nums[lens-i], nums[lens-j] = nums[lens-j], nums[lens-i]
                        flag = True
                    break
                elif nums[lens-i] == nums[lens-j]:
                    break
                else:
                    if not partial and nums[lens-j+1] > nums[lens-j]:
                        partial = True

        if not flag:
            nums.reverse()
            return nums
        st = lens-j+1
        ed = lens-1
        #print(nums,st,ed)

        '''
        while st < ed and nums[st] > nums[ed]:
            nums[st],nums[ed] = nums[ed],nums[st]
            st += 1
            ed -= 1
        '''
        for i in range(st, ed):
            for j in range(i, ed+1):
                if nums[i] > nums[j]:
                    nums[i],nums[j] = nums[j],nums[i]
        return nums
        