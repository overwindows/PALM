class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        val = x^y
        dist = 0
        while val:
            val = val & (val-1)
            dist += 1
        return dist