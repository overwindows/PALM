class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        xor_lst = [2**i for i in range(n)]
        
        res = []
        for x in range(2**n):
            if len(res) == 0:
                res.append(x)
            else:
                prev = res[-1]
                for e in xor_lst:
                    if prev^e in res:
                        continue
                    else:
                        res.append(prev^e)
                        break
        
        return res
                        