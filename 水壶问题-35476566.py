class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x < y:
            x,y = y,x
        
        if z == 0:
            return True
        
        if z > x+y:
            return False
        
        if y == 0:
            return x == z
        
        if x == y and x != z:
            return False
        
        if (x+z) % y == 0:
            return True
        
        if z % (x-y) == 0:
            return True
        
        if (z+y)%(x-y) == 0:
            return True
        
        gcd = y
        #print(x,y)
        while x % y:
            x, y = y, x%y
            #print(x,y)
        
        return z%y == 0
        
        # return False