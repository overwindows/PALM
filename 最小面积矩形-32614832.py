#import sys
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        points.sort()
        if len(points) < 4:
            return 0
        lens = len(points)
        #print(lens)
        bitmap = set()
        hashmap = {}
        min_area =  sys.maxsize
        _x = set()
        for x,y in points:
            bitmap.add((x,y))
            if x in hashmap:
                hashmap[x].append(y)
            else:
                hashmap[x] = [y]
            _x.add(x)
        
        if len(_x) == 1:
            return 0        
        
        _x = list(_x)
        _x.sort()
        if lens > 100:
            #print(_x)
            for i in range(len(_x)-1):
                for j in range(i+1,len(_x)):
                    _y = hashmap[_x[i]]
                    _y.sort()
                    for k in range(len(_y)-1):
                        for l in range(k+1,len(_y)):
                            if (_x[j],_y[k]) in bitmap and (_x[j],_y[l]) in bitmap:
                                area = (_x[j] - _x[i]) * (_y[l] - _y[k])
                                if area > 0 and area < min_area:
                                    min_area = area                      
        else:    
            for i in range(lens-3):
                for j in range(i+1, lens-2):
                    if points[i][0] != points[j][0]:
                        continue
                    for k in range(j+1, lens-1):
                        if points[j][0] == points[k][0]:
                            continue
                        if (points[k][0],points[i][1]) in bitmap and (points[k][0],points[j][1]) in bitmap:
                            area = (points[k][0] - points[i][0]) * (points[j][1] - points[i][1])
                            if area > 0 and area < min_area:
                                min_area = area

        
        if min_area == sys.maxsize:
            return 0
        
        return min_area