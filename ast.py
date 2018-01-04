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

class Variable(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Variable"

    def accept(self, visitor):
        return visitor.visit_variable(self)

class Expression(object):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return "Expression"

    def accept(self, visitor):
        return visitor.visit_expression(self)

class Print(object):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return "Print"

    def accept(self, visitor):
        return visitor.visit_print(self)

class Var(object):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer

    def __str__(self):
        return "Var"

    def accept(self, visitor):
        return visitor.visit_var(self)

class Assign(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return "Assign"

    def accept(self, visitor):
        return visitor.visit_assign(self)

class Block(object):
    def __init__(self, statements):
        self.statements = statements

    def __str__(self):
        return "Block"

    def accept(self, visitor):
        return visitor.visit_block(self)

class If(object):
    def __init__(self, condition, then_branch, else_branch):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __str__(self):
        return "If"

    def accept(self, visitor):
        return visitor.visit_if(self)

class Logical(object):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return "Logical"

    def accept(self, visitor):
        return visitor.visit_logical(self)

class While(object):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __str__(self):
        return "While"

    def accept(self, visitor):
        return visitor.visit_while(self)

class Call(object):
    def __init__(self, callee, paren, arguments):
        self.callee = callee
        self.paren = paren
        self.arguments = arguments

    def __str__(self):
        return "Call"

    def accept(self, visitor):
        return visitor.visit_call(self)

class Function(object):
    def __init__(self, name, parameters, body):
        self.name = name
        self.parameters = parameters
        self.body = body

    def __str__(self):
        return "Function"

    def accept(self, visitor):
        return visitor.visit_function(self)

