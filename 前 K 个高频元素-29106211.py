class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hash_tbl = {}
        for num in nums:
            if num in hash_tbl:
                hash_tbl[num] += 1
            else:
                hash_tbl[num] = 1
        
        heap = []
        ret = heapq.nlargest(k, hash_tbl.items(), key=lambda x:x[1])
        res = [x for x,y in ret]
        
        return res
        
        