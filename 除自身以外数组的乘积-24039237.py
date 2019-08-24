class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prod_fwd = [1] * len(nums)
        prod_bwd = [1] * len(nums)
        prods = [0] * len(nums)
        
        #prod_fwd[0] = nums[0]
        #prod_bwd[0] = nums[-1]
        for i in range(1,len(nums)):
            prod_fwd[i] = prod_fwd[i-1] * nums[i-1]
            prod_bwd[i] = prod_bwd[i-1] * nums[-i]
        #print(prod_fwd, prod_bwd)
        
        for i in range(len(nums)):
            prods[i] = prod_fwd[i] * prod_bwd[-(i+1)]
        return prods