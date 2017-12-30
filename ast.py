class Binary(object):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return "Binary"

    def accept(self, visitor):
        return visitor.visit_binary(self)

class Grouping(object):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return "Grouping"

    def accept(self, visitor):
        return visitor.visit_grouping(self)

class Literal(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Literal"

    def accept(self, visitor):
        return visitor.visit_literal(self)

class Unary(object):
    def __init__(self, operator, right):
        self.operator = operator
        self.right = right

    def __str__(self):
        return "Unary"

    def accept(self, visitor):
        return visitor.visit_unary(self)

