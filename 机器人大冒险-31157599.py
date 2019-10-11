class Solution:
    def robot(self, command: str, obstacles: List[List[int]], x: int, y: int) -> bool:
        rect_x = 0
        rect_y = 0
        
        if len(command) == 0:
            return False
        
        for c in command:
            if c == 'U':
                rect_y += 1
            elif c == 'R':
                rect_x += 1
        
        for o in obstacles:
            o_x, o_y = o
            #print(o_x,o_y)
            if o_x < x or o_y < y:
                if rect_x > 0:
                    n = o_x // rect_x
                else:
                    n = o_y // rect_y
                
                o_x0 = rect_x * n
                o_y0 = rect_y * n

                o_x1 = rect_x * (n+1)
                o_y1 = rect_y * (n+1)
                
                #print(o_x0,o_y0,o_x1,o_y1)

                if o_x >= o_x0 and o_x <= o_x1 and o_y >= o_y0 and o_y <= o_y1:
                    start_x = o_x0
                    start_y = o_y0
                    
                    if start_x == o_x and start_y == o_y:
                        return False
                    
                    for c in command:
                        if c == 'U':
                            start_y += 1
                        elif c == 'R':
                            start_x += 1

                        if start_x == o_x and start_y == o_y:
                            return False
                    
                    #if o_x1 == o_x and o_y1 == o_y:
                    #    return False
                else:
                    pass
        
        if rect_x > 0:
            n = x // rect_x
        else:
            n = y // rect_y
        rect_x0 = rect_x * n
        rect_y0 = rect_y * n
        
        rect_x1 = rect_x * (n+1)
        rect_y1 = rect_y * (n+1)
        
        if x >= rect_x0 and x <= rect_x1 and y >= rect_y0 and y <= rect_y1:
            start_x = rect_x0
            start_y = rect_y0
            
            if start_x == x and start_y == y:
                return True
            
            for c in command:
                if c == 'U':
                    start_y += 1
                elif c == 'R':
                    start_x += 1
                
                if start_x == x and start_y == y:
                    return True
            return False
        else:
            return False