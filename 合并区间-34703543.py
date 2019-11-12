class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        #print(intervals)
        if len(intervals) < 2:
            return intervals
        merge = []
        for interval in intervals:
            if merge:
                s1,e1 = interval
                s0,e0 = merge[-1]
                if s1 <= e0:
                    merge.pop()
                    if e1 > e0:
                        merge.append([s0,e1])
                    else:
                        merge.append([s0,e0])
                else:
                    merge.append([s1,e1])
            else:
                merge.append(interval)
        return merge