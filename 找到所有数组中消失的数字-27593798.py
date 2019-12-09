class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        missing = []
        
        for i in range(len(nums)):
            ix = i
            while nums[ix] != ix+1:
                #print(nums)
                if nums[nums[ix]-1] == nums[ix]:
                    break  
                tmp = nums[nums[ix]-1]
                nums[nums[ix]-1] = nums[ix]
                nums[ix] = tmp
        
        for i in range(len(nums)):
            if nums[i] != i+1:
                missing.append(i+1)
        
        return missing
            

                    
                
                

        
        