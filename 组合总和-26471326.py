class Solution(object):
    
    def combSum(self, candidates, target, path):
        if target == 0:
            return [path]
        
        if target < 0:
            return []
        
        res = []
        for i in range(len(candidates)):
            val = candidates[i]
            _candidates = candidates[i:]
            _path = path[:]
            _path.append(val)
            _res = self.combSum(_candidates, target-val, _path)
            res.extend(_res)
        
        return res
    
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """ 
        candidates.sort()
        candidates.reverse()
        res = []
        for i in range(len(candidates)):
            val = candidates[i]
            _candidates = candidates[i:]
            _res = self.combSum(_candidates, target-val, [val])
            res.extend(_res)
        
        return res