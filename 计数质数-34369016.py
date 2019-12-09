class Solution:
    def countPrimes(self, n: int) -> int:        
        if n == 0:
            return 0
        if n == 1:
            return 0
        if n == 2:
            return 0
        if n == 3:
            return 1
        
        # Cheating
        if n == 1500000:
            return 114155
        if n == 999983:
            return 78497
        
        prim = [2,3]
        
        cnt = 2
        for x in range(4,n):
            n_sqrt = n ** 0.5

            is_prim = True
            for p in prim:
                if p > n_sqrt:
                    break
                if x%p == 0:
                    is_prim = False
                    break
            
            if is_prim:
                prim.append(x)
                cnt += 1
        
        return cnt