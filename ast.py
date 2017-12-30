class Binary(object):
    def __init__(left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return "Binary"

    def visit(self, visitor):
        return visitor.visit_binary(self)

class Grouping(object):
    def __init__(expression):
        self.expression = expression

    def __str__(self):
        return "Grouping"

    def visit(self, visitor):
        return visitor.visit_grouping(self)

class Literal(object):
    def __init__(value):
        self.value = value

    def __str__(self):
        return "Literal"

    def visit(self, visitor):
        return visitor.visit_literal(self)

class Unary(object):
    def __init__(operator, right):
        self.operator = operator
        self.right = right

    def __str__(self):
        return "Unary"

    def visit(self, visitor):
        return visitor.visit_unary(self)

