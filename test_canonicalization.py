from Shane import canonicalize


def test1():
    net1 = "X : XY : E\nY : X"
    net2 = "Y : X\nX : XY : E"
    assert (canonicalize(net1) == canonicalize(net2))


N1 = np.array([[1, 1], [1, 0]])
N2 = np.array([[0, 1], [1, 1]])


def test2():
    net1 = "X : XY : E\nY : X"
    net2 = "Y : X : E\nX : XY"
    assert (canonicalize(net1) != canonicalize(net2))


def test3():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)"
    net2 = "X : XY\nY : Y(Z+X)\nZ : (~X)(~Y)"
    assert (canonicalize(net1) == canonicalize(net2))


def test4():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)"
    net2 = "Y : Y(Z+X)\nZ : (~X)(~Y)\nX : XY"
    assert (canonicalize(net1) == canonicalize(net2))


def test5():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)"
    net2 = "Y : Y(Z+X) : E\nZ : (~X)(~Y) : E\nX : XY : E"
    assert (canonicalize(net1) != canonicalize(net2))


def test6():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)"
    net2 = "Y : Z(Y+X)\nZ : (~X)(~Y)\nX : XY"
    assert (canonicalize(net1) != canonicalize(net2))


def test7():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    net2 = "X : XY\nY : (X+~Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    assert (canonicalize(net1) != canonicalize(net2))


def test8():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    net2 = "V:(~U)(X+V)\nX : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V"
    assert (canonicalize(net1) == canonicalize(net2))


def test9():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    net2 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : (Z)(V)\nV:(X+V)(~U)"
    assert (canonicalize(net1) != canonicalize(net2))


def test10():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    net2 = "X : XY\nY : (X)(Z)(Y)\nZ : (~X)(~Y)\nU : (Z)(V)\nV:(X+V)(~U)"
    assert (canonicalize(net1) != canonicalize(net2))


def test11():
    net1 = "X : XY\nY : (X+Z)Y\nZ : (~X)(~Y)\nU : Z+V\nV:(X+V)(~U)"
    net2 = "X : XY\nY : (~X)(Z)(Y)\nZ : (~X)(~Y)\nU : (Z)(V)\nV:(X+V)(~U)"
    assert (canonicalize(net1) != canonicalize(net2))
