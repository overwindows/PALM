class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        
        if len(nums) == 1:
          return 0
      
        org_nums = nums[:]
        nums.sort()
        
        st = 0
        ed = len(nums) - 1
        
        move = True
        
        while move and st < ed:
          move = False
          if org_nums[st] == nums[st]:
            st+=1
            move = True
          
          if org_nums[ed] == nums[ed]:
            ed-=1
            move = True
        
        if ed <= st:
          return 0
        
        return ed-st+1
      