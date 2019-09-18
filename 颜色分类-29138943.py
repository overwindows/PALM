class Solution:
    
    def sortColorsEx(self, nums, st, ed):
        if st == ed:
            return 
        #print(st,ed)
        while st < len(nums) and nums[st] == 0:
            st += 1
        
        if st == len(nums):
            return
        
        while ed > -1 and nums[ed] == 2:
            ed -= 1
        
        if ed == -1:
            return
        
        if st > ed:
            return 
                
        if nums[st] == 2 and nums[ed] == 0:
            nums[st], nums[ed] = nums[ed], nums[st]
            if st+1 < len(nums): 
                self.sortColorsEx(nums,st+1,ed-1)
        
        elif nums[st] == 2 and nums[ed] == 1:
            nums[st], nums[ed] = nums[ed], nums[st]
            self.sortColorsEx(nums,st,ed-1)
        
        elif nums[st] == 1 and nums[ed] == 0:
            nums[st], nums[ed] = nums[ed], nums[st]
            if st+1 < ed and ed < len(nums):
                self.sortColorsEx(nums,st+1,ed)
        
        elif nums[st] == 1 and nums[ed] == 1 and st!=ed:
            if st+1 == ed: # [1,1]
                return
            
            if st+1 == ed-1: # [1,0/2,1]
                if nums[st+1] == 0:
                    nums[st+1] ,nums[st] = nums[st], nums[st+1]
                elif nums[st+1] == 2:
                    nums[ed] ,nums[ed-1] = nums[ed-1], nums[ed]
                return
            
            #print(st+1,ed)
            _st = st
            while _st < ed:
                if nums[_st] != 1:
                    break
                else:
                    _st += 1
            if _st == ed:
                return 
            nums[_st] ,nums[st] = nums[st], nums[_st]
            self.sortColorsEx(nums,st,ed)
            #nums[st+1] ,nums[st] = nums[st], nums[st+1]
            #nums[ed] ,nums[ed-1] = nums[ed-1], nums[ed]
            
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 0:
            return []
        
        if len(nums) == 1:
            return nums
                
        st = 0
        ed = len(nums)-1

        self.sortColorsEx(nums,st,ed)

            
        
        
        