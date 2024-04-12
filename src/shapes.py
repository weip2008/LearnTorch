def drawCircle(pen1, x, y, r):
    """
    draw a circle with radius r and position (x, y)
    """
    pen1.penup()
    pen1.goto(x,y)
    pen1.down()
    pen1.circle(r)
    pen1.seth(0)
    pen1.penup()

def drawRectangle(pen1, x, y, width, height):
    """
    draw a circle with radius r and position (x, y)
    """
    pen1.up()
    pen1.goto(x, y)
    pen1.down()
    pen1.fd(width)
    pen1.left(90)
    pen1.fd(height)
    pen1.left(90)
    pen1.fd(width)
    pen1.left(90)
    pen1.fd(height)
    pen1.seth(0)

def drawLine(pen1, x,y,angle,length):
    """
    draw a stright line start at position (x, y), turn right by angle, and forward with length.
    """
    pen1.up()
    pen1.goto(x,y)
    pen1.down()
    pen1.left(angle)
    pen1.fd(length)
    pen1.seth(0)

def drawEquilateral(pen1, x, y,angle,side):
    """
    draw a triangle at position (x, y), turn right by angle, with each side.
    """
    pen1.up()
    pen1.goto(x,y)
    pen1.down()
    pen1.left(angle)
    pen1.fd(side)
    pen1.left(120)
    pen1.fd(side)
    pen1.left(120)
    pen1.fd(side)
    pen1.seth(0)
    
def drawEllipse(pen1, x, y, radius, angle):
    pen1.up()
    pen1.goto(x,y)
    pen1.down()
    for i in range(2): 
        # two arcs 
        pen1.circle(radius,90) 
        pen1.circle(radius//2,90) 
    pen1.seth(0)

def drawText(pen1, x, y, text):
    pen1.up()
    pen1.goto(x,y)
    pen1.down()
    pen1.write(text)
    pen1.seth(0)

def drawTriangle(pen1, p1, p2, p3):
    pen1.penup()
    pen1.goto(p1)
    pen1.pendown()
    pen1.goto(p2)
    pen1.goto(p3)
    pen1.goto(p1)
    pen1.seth(0)

def drawLine2(pen1, p1, p2):
    pen1.penup()
    pen1.goto(p1)
    pen1.pendown()
    pen1.goto(p2)
    pen1.seth(0)