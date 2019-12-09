class Solution:

    def __init__(self, radius: float, x_center: float, y_center: float):
        self.r = radius
        self.x = x_center
        self.y = y_center
    

    def randPoint(self) -> List[float]:
        while True:
            x = random.uniform(-self.r,self.r)
            y = random.uniform(-self.r,self.r)
            if x**2 + y**2 <= self.r**2:
                return [self.x+x,self.y+y]


# Your Solution object will be instantiated and called as such:
# obj = Solution(radius, x_center, y_center)
# param_1 = obj.randPoint()