#!/usr/bin/env python3

import sys
import readline # noqa
from enum import Enum, auto
import ast


class TokenType(Enum):
    # Single-character tokens
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    MINUS = auto()
    PLUS = auto()
    SEMICOLON = auto()
    SLASH = auto()
    STAR = auto()

    # One or two character tokens
    BANG = auto()
    BANG_EQUAL = auto()
    EQUAL = auto()
    EQUAL_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # Keywords
    AND = auto()
    CLASS = auto()
    ELSE = auto()
    FALSE = auto()
    FUN = auto()
    FOR = auto()
    IF = auto()
    NIL = auto()
    OR = auto()
    PRINT = auto()
    RETURN = auto()
    SUPER = auto()
    THIS = auto()
    TRUE = auto()
    VAR = auto()
    WHILE = auto()

    EOF = auto()


KEYWORDS = {
        'and': TokenType.AND,
        'class': TokenType.CLASS,
        'else': TokenType.ELSE,
        'false': TokenType.FALSE,
        'for': TokenType.FOR,
        'fun': TokenType.FUN,
        'if': TokenType.IF,
        'nil': TokenType.NIL,
        'or': TokenType.OR,
        'print': TokenType.PRINT,
        'return': TokenType.RETURN,
        'super': TokenType.SUPER,
        'this': TokenType.THIS,
        'true': TokenType.TRUE,
        'var':  TokenType.VAR,
        'while': TokenType.WHILE,
}


class Token(object):
    def __init__(self, ttype, lexeme, literal, line):
        self.ttype = ttype
        self.lexeme = str(lexeme)
        self.literal = literal
        self.line = int(line)

    def __str__(self):
        return '({ttype!s}, "{lexeme!s}", {literal!s})'.format(
                ttype=self.ttype, lexeme=self.lexeme, literal=self.literal)

    def __repr__(self):
        return self.__str__()


class LoxFunction(object):
    def __init__(self, declaration, closure, is_initializer):
        self.declaration = declaration
        self.closure = closure
        self.is_initializer = is_initializer

    def arity(self):
        return len(self.declaration.parameters)

    def call(self, interp, args):
        env = Environment(self.closure)
        params = self.declaration.parameters
        for param, arg in zip(params, args):
            env.define(param.lexeme, arg)
        try:
            interp.execute_block(self.declaration.body, env)
        except ReturnException as rv:
            return rv.value
        if self.is_initializer:
            return self.closure.get_at(0, "this")

    def bind(self, instance):
        environment = Environment(self.closure)
        environment.define("this", instance)
        return LoxFunction(self.declaration, environment, self.is_initializer)

    def __str__(self):
        return "<fn {}>".format(self.declaration.name.lexeme)


class LoxClass(object):
    def __init__(self, name, superclass, methods):
        self.name = name
        self.methods = methods
        self.superclass = superclass

    def __str__(self):
        return self.name

    def call(self, interp, args):
        instance = LoxInstance(self)
        initializer = self.methods.get("init")
        if initializer is not None:
            initializer.bind(instance).call(interp, args)
        return instance

    def arity(self):
        initializer = self.methods.get("init")
        if initializer is not None:
            return initializer.arity()
        else:
            return 0

    def find_method(self, instance, name):
        try:
            return self.methods[name].bind(instance)
        except KeyError:
            return None


class LoxInstance(object):
    def __init__(self, klass):
        self.klass = klass
        self.fields = {}

    def __str__(self):
        return '{!s} instance'.format(self.klass)

    def get(self, name):
        if name.lexeme in self.fields:
            return self.fields[name.lexeme]
        method = self.klass.find_method(self, name.lexeme)
        if method is not None:
            return method
        else:
            # XXX: error handling
            raise RuntimeError("Undefined property '{}'.".format(
                name.lexeme))

    def set(self, name, value):
        self.fields[name.lexeme] = value


class Builtin_clock(object):
    def __init__(self):
        pass

    def arity(self):
        return 0

    def call(self, interp, args):
        import time
        return time.clock()


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'ReturnException'


class Interpreter(object):
    def __init__(self):
        self.had_error = False
        self._globals = Environment(name="globals")
        self._globals.define("clock", Builtin_clock())
        self.environment = self._globals
        self._locals = {}  # Expr -> Integer

    def interpret(self, statements):
        self.had_error = False
        try:
            for stmt in statements:
                self.execute(stmt)
        except Exception as e:
            lox_runtime_error(e)
            # TEMP TEMP
            raise

    def execute(self, stmt):
        stmt.accept(self)

    def visit_expression(self, stmt):
        self.evaluate(stmt.expression)

    def visit_print(self, stmt):
        value = self.evaluate(stmt.expression)
        print(str(value))

    def visit_this(self, expr):
        return self.lookup_variable(expr.keyword, expr)

    def visit_get(self, expr):
        object = self.evaluate(expr.object)
        if not isinstance(object, LoxInstance):
            # XXX: error handling
            raise RuntimeError("Only instances have properties.")
        return object.get(expr.name)

    def visit_set(self, expr):
        object = self.evaluate(expr.object)
        if not isinstance(object, LoxInstance):
            # XXX: error handling
            raise RuntimeError("Only instances have fields.")
        value = self.evaluate(expr.value)
        object.set(expr.name, value)
        return value

    def visit_if(self, stmt):
        if Interpreter.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch is not None:
            self.execute(stmt.else_branch)

    def visit_return(self, stmt):
        if stmt.value is None:
            value = None
        else:
            value = self.evaluate(stmt.value)
        raise ReturnException(value)

    def evaluate(self, expr):
        return expr.accept(self)

    def visit_binary(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        op = expr.operator.ttype
        if op == TokenType.MINUS:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) - float(right)
        elif op == TokenType.PLUS:
            if isinstance(left, float) and isinstance(right, float):
                Interpreter.check_number_operands(op, left, right)
                result = float(left) + float(right)
            elif isinstance(left, str) and isinstance(right, str):
                result = str(left) + str(right)
            else:
                raise ValueError("Operands must be two numbers or two "
                        "strings.")
        elif op == TokenType.SLASH:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) / float(right)
        elif op == TokenType.STAR:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) * float(right)
        elif op == TokenType.GREATER:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) > float(right)
        elif op == TokenType.GREATER_EQUAL:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) >= float(right)
        elif op == TokenType.LESS:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) < float(right)
        elif op == TokenType.LESS_EQUAL:
            Interpreter.check_number_operands(op, left, right)
            result = float(left) <= float(right)
        elif op == TokenType.BANG_EQUAL:
            result = not Interpreter.is_equal(left, right)
        elif op == TokenType.EQUAL_EQUAL:
            result = Interpreter.is_equal(left, right)
        else:
            # XXX: raise error?
            result = None
        return result

    def visit_grouping(self, expr):
        return self.evaluate(expr.expression)

    def visit_literal(self, expr):
        return expr.value

    def visit_unary(self, expr):
        right = self.evaluate(expr.right)
        op = expr.operator.ttype
        if op == TokenType.MINUS:
            Interpreter.check_number_operand(op, right)
            return -1. * float(right)
        elif op == TokenType.BANG:
            return not Interpreter.is_truthy(right)
        else:
            # XXX: should this raise an Exception?  Shouldn't ever happen
            #      right?
            return None

    def visit_block(self, stmt):
        self.execute_block(stmt.statements, Environment(self.environment))

    def execute_block(self, statements, environment):
        previous = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous

    @staticmethod
    def is_truthy(val):
        if val is None:
            return False
        elif isinstance(val, bool):
            return val
        else:
            # XXX: seems like numbers should return "val != 0."
            return True

    @staticmethod
    def is_equal(left, right):
        if left is None and right is None:
            return True
        elif left is None:
            return False
        else:
            return left == right

    @staticmethod
    def check_number_operand(operator, operand):
        if not isinstance(operand, float):
            raise ValueError("Operand must be a number: {op!s}".format(
                op=operator))

    def check_number_operands(operator, left, right):
        if not isinstance(left, float):
            raise ValueError("Left operand must be a number: {op!s}".format(
                operator))
        elif not isinstance(right, float):
            raise ValueError("Right operand must be a number: {op!s}".format(
                operator))

    def visit_var(self, stmt):
        value = None
        if stmt.initializer is not None:
            value = self.evaluate(stmt.initializer)
        self.environment.define(stmt.name.lexeme, value)

    def visit_variable(self, expr):
        # return self.environment.get(expr.name)
        return self.lookup_variable(expr.name, expr)

    def lookup_variable(self, name, expr):
        distance = self._locals.get(expr)
        if distance is not None:
            return self.environment.get_at(distance, name.lexeme)
        else:
            return self._globals.get(name)

    def visit_assign(self, expr):
        value = self.evaluate(expr.value)
        # self.environment.assign(expr.name, value)
        distance = self._locals.get(expr)
        if distance is not None:
            self.environment.assign_at(distance, expr.name, value)
        else:
            self._globals.assign(expr.name, value)
        return value

    def visit_logical(self, expr):
        left = self.evaluate(expr.left)
        if expr.operator.ttype == TokenType.OR:
            if Interpreter.is_truthy(left):
                return left
        else:
            if not Interpreter.is_truthy(left):
                return left
        return self.evaluate(expr.right)

    def visit_while(self, stmt):
        while Interpreter.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

    def visit_call(self, expr):
        callee = self.evaluate(expr.callee)
        arguments = [self.evaluate(arg) for arg in expr.arguments]
        # # TODO(plesslie): check that callee is a callable
        # if not isinstance(callee, XXX):
        #     raise RuntimeError("Can only call functions and classes.")
        if len(arguments) != callee.arity():
            raise RuntimeError("Expected {} arguments but got {}.".format(
                callee.arity(), len(arguments)))
        return callee.call(self, arguments)

    def visit_function(self, stmt):
        function = LoxFunction(stmt, self.environment, False)
        self.environment.define(stmt.name.lexeme, function)

    def visit_class(self, stmt):
        self.environment.define(stmt.name.lexeme, None)
        if stmt.superclass is not None:
            superclass = self.evaluate(stmt.superclass)
            if not isinstance(superclass, LoxClass):
                raise RuntimeError("Superclass must be a class.")
        else:
            superclass = None
        methods = {}
        for method in stmt.methods:
            is_init = method.name.lexeme == "init"
            function = LoxFunction(method, self.environment, is_init)
            methods[method.name.lexeme]  = function
        klass = LoxClass(stmt.name.lexeme, superclass, methods)
        self.environment.assign(stmt.name, klass)

    def resolve(self, expr, depth):
        self._locals[expr] = depth


class FunctionType(Enum):
    NONE = auto(),
    FUNCTION = auto(),
    INITIALIZER = auto(),
    METHOD = auto(),


class ClassType(Enum):
    NONE = auto(),
    CLASS = auto(),


class Resolver(object):
    def __init__(self, interp):
        self.interp = interp
        self.scopes = []
        self.current_function = FunctionType.NONE
        self.current_class = ClassType.NONE

    def resolve(self, stmt):
        if isinstance(stmt, list):
            for s in stmt:
                s.accept(self)
        else:
            stmt.accept(self)

    def begin_scope(self):
        self.scopes.append({})

    def end_scope(self):
        self.scopes.pop()

    def visit_block(self, stmt):
        # XXX: make this a context manager?
        self.begin_scope()
        self.resolve(stmt.statements)
        self.end_scope()

    def visit_get(self, expr):
        self.resolve(expr.object)

    def visit_set(self, expr):
        self.resolve(expr.value)
        self.resolve(expr.object)

    def visit_this(self, expr):
        if self.current_class == ClassType.NONE:
            # XXX: error handling
            raise RuntimeError("Cannot use 'this' outside of a class.")
        self.resolve_local(expr, expr.keyword)

    def visit_class(self, stmt):
        self.declare(stmt.name)
        self.define(stmt.name)
        enclosing_class = self.current_class
        self.current_class = ClassType.CLASS
        if stmt.superclass is not None:
            self.resolve(stmt.superclass)
        self.begin_scope()
        self.scopes[-1]['this'] = True
        for method in stmt.methods:
            if method.name.lexeme == "init":
                decl = FunctionType.INITIALIZER
            else:
                decl = FunctionType.METHOD
            self.resolve_function(method, decl)
        self.end_scope()
        self.current_class = enclosing_class

    def visit_var(self, stmt):
        self.declare(stmt.name)
        if stmt.initializer is not None:
            self.resolve(stmt.initializer)
        self.define(stmt.name)

    def declare(self, name):
        if not self.scopes:
            return
        scope = self.scopes[-1]
        if name.lexeme in scope:
            # XXX: error handling function
            raise ValueError("Variable with this name already declared in this scope.")
        scope[name.lexeme] = False

    def define(self, name):
        if not self.scopes:
            return
        self.scopes[-1][name.lexeme] = True

    def visit_variable(self, expr):
        if self.scopes:
            try:
                if not self.scopes[-1][expr.name.lexeme]:
                    # XXX: what error method to call?
                    # error(expr.name, "Cannot read local variable in its own initializer.")
                    raise RuntimeError("Cannot read local variable in its own initializer.")
            except KeyError:
                pass
        self.resolve_local(expr, expr.name)

    def resolve_local(self, expr, name):
        for i, scope in enumerate(reversed(self.scopes)):
            if name.lexeme in scope:
                self.interp.resolve(expr, i)
                return
        # Not found. Assume it is global

    def visit_assign(self, expr):
        self.resolve(expr.value)
        self.resolve_local(expr, expr.name)

    def visit_function(self, stmt):
        self.declare(stmt.name)
        self.define(stmt.name)
        self.resolve_function(stmt, FunctionType.FUNCTION)

    def resolve_function(self, function, ftype):
        enclosing_function = self.current_function
        self.current_function = ftype
        self.begin_scope()
        for param in function.parameters:
            self.declare(param)
            self.define(param)
        self.resolve(function.body)
        self.end_scope()
        self.current_function = enclosing_function

    def visit_expression(self, stmt):
        self.resolve(stmt.expression)

    def visit_if(self, stmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.then_branch)
        if stmt.else_branch:
            self.resolve(stmt.else_branch)

    def visit_print(self, stmt):
        self.resolve(stmt.expression)

    def visit_return(self, stmt):
        if self.current_function == FunctionType.NONE:
            # XXX: error handling
            raise RuntimeError("Cannot return from top-level code.")
        if stmt.value is not None:
            if self.current_function == FunctionType.INITIALIZER:
                # XXX: error handling
                raise RuntimeError("Cannot return a value from an initializer.")
            self.resolve(stmt.value)

    def visit_while(self, stmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.body)

    def visit_binary(self, expr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def visit_call(self, expr):
        self.resolve(expr.callee)
        for arg in expr.arguments:
            self.resolve(arg)

    def visit_grouping(expr):
        self.resolve(expr.expression)

    def visit_literal(self, expr):
        pass

    def visit_logical(self, expr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def visit_unary(self, expr):
        self.resolve(expr.right)


def lox_runtime_error(err):
    # TODO: make exception class to hold line number
    print(str(err))


def interpret(expr):
    try:
        interp = Interpreter()
        ok, result = interp.run(expr)
        # XXX: doesn't seem like this function should be stateful...
        print(str(result))
    except Exception as e:
        lox_runtime_error(e)


# REVISIT: this is a class in the text, re-eval if this needs to
# be a class or that is just Java dumb-ness
def scan_tokens(source):
    tokens = []
    start = 0
    current = 0
    line = 1
    end = len(source)

    def add_token(ttype, literal=None):
        lexeme = source[start:current]
        tokens.append(Token(ttype, lexeme, literal, line))

    def advance():
        nonlocal current
        result = source[current]
        current += 1
        return result

    def match(expected):
        nonlocal current
        if not current < end:
            return False
        if source[current] != expected:
            return False
        current += 1
        return True

    def peek():
        if not current < end:
            return '\0'
        return source[current]

    def peek_next():
        if not current + 1 < end:
            return '\0'
        return source[current + 1]

    while current < end:
        start = current
        c = advance()
        if c == '(':
            add_token(TokenType.LEFT_PAREN)
        elif c == ')':
            add_token(TokenType.RIGHT_PAREN)
        elif c == '{':
            add_token(TokenType.LEFT_BRACE)
        elif c == '}':
            add_token(TokenType.RIGHT_BRACE)
        elif c == ',':
            add_token(TokenType.COMMA)
        elif c == '.':
            add_token(TokenType.DOT)
        elif c == '-':
            add_token(TokenType.MINUS)
        elif c == '+':
            add_token(TokenType.PLUS)
        elif c == ';':
            add_token(TokenType.SEMICOLON)
        elif c == '*':
            add_token(TokenType.STAR)
        elif c == '!':
            add_token(TokenType.BANG_EQUAL if match('=') else TokenType.BANG)
        elif c == '=':
            add_token(TokenType.EQUAL_EQUAL if match('=') else TokenType.EQUAL)
        elif c == '<':
            add_token(TokenType.LESS_EQUAL if match('=') else TokenType.LESS)
        elif c == '>':
            add_token(TokenType.GREATER_EQUAL if match('=') else TokenType.GREATER)
        elif c == '/':
            if match('/'):
                # A comment goes until the end of the line.
                while peek() != '\n' and current < end:
                    advance()
            else:
                add_token(TokenType.SLASH)
        elif c in (' ', '\r', '\t'):
            continue
        elif c == '\n':
            line += 1
            continue
        elif c == '"':
            while peek() != '"' and current < end:
                if peek() == '\n':
                    line += 1
                advance()
            # unterminated string
            if not current < end:
                # XXX: text uses error()
                raise ValueError("Unterminated string")

            # The closing ".
            advance()

            # Trim the surrounding quotes
            value = source[start+1:current-1]
            add_token(TokenType.STRING, value)
        elif c.isdigit():
            while peek().isdigit():
                advance()
            if peek() == '.' and peek_next().isdigit():
                advance()
                while peek().isdigit():
                    advance()
            add_token(TokenType.NUMBER, float(source[start:current]))
        elif c.isalpha():
            while peek().isalnum():
                advance()
            text = source[start:current]
            try:
                ttype = KEYWORDS[text]
            except KeyError:
                ttype = TokenType.IDENTIFIER
            add_token(ttype)
        else:
            # XXX: text uses global for marking an error, exception seems
            # more natural at this point
            raise ValueError("[line {}] Unrecognized character: '{}'".format(
                line, c))
            # error(line, "Unexpected character.")

    tokens.append(Token(TokenType.EOF, '', None, line))
    return tokens


class Environment(object):
    def __init__(self, enclosing=None, name=None):
        self.values = {}
        self.enclosing = enclosing
        self.name = name

    def define(self, name, value):
        self.values[name] = value

    def get(self, name):
        try:
            return self.values[name.lexeme]
        except KeyError:
            pass
        if self.enclosing is not None:
            return self.enclosing.get(name)
        else:
            raise RuntimeError("Undefined variable '{}'.".format(
                name.lexeme))

    def get_at(self, distance, name):
        return self.ancestor(distance).values[name]

    def ancestor(self, distance):
        result = self
        for i in range(distance):
            result = result.enclosing
        return result

    def assign(self, name, value):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
        elif self.enclosing is not None:
            self.enclosing.assign(name, value)
        else:
            raise RuntimeError("Undefined variable '{}'.".format(
                name.lexeme))

    def assign_at(self, distance, name, value):
        self.ancestor(distance).values[name.lexeme] = value


class Parser(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def peek(self):
        return self.tokens[self.current]

    def check(self, ttype):
        if self.peek().ttype == TokenType.EOF:
            return False
        else:
            return self.peek().ttype == ttype

    def match(self, *args):
        for arg in args:
            if self.check(arg):
                self.advance()
                return True
        return False

    def advance(self):
        if self.peek().ttype != TokenType.EOF:
            self.current += 1
        return self.previous()

    def previous(self):
        return self.tokens[self.current - 1]

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.or_()
        if self.match(TokenType.EQUAL):
            equals = self.previous()
            value = self.assignment()
            if isinstance(expr, ast.Variable):
                name = expr.name
                return ast.Assign(name, value)
            elif isinstance(expr, ast.Get):
                return ast.Set(expr.object, expr.name, value)
            else:
                self.error(equals, "Invalid assignment target.")
        return expr

    def or_(self):
        expr = self.and_()
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.and_()
            expr = ast.Logical(expr, operator, right)
        return expr

    def and_(self):
        expr = self.equality()
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = ast.Logical(expr, operator, right)
        return expr

    def equality(self):
        expr = self.comparison()
        while self.match(TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = ast.Binary(expr, operator, right)
        return expr

    def comparison(self):
        expr = self.addition()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL,
                TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous()
            right = self.addition()
            expr = ast.Binary(expr, operator, right)
        return expr

    def addition(self):
        expr = self.multiplication()
        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous()
            right = self.multiplication()
            expr = ast.Binary(expr, operator, right)
        return expr

    def multiplication(self):
        expr = self.unary()
        while self.match(TokenType.SLASH, TokenType.STAR):
            operator = self.previous()
            right = self.unary()
            expr = ast.Binary(expr, operator, right)
        return expr

    def unary(self):
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return ast.Unary(operator, right)
        else:
            return self.call()

    def call(self):
        expr = self.primary()
        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER,
                        "Expect property name after '.'.")
                expr = ast.Get(expr, name)
            else:
                break
        return expr

    def finish_call(self, callee):
        arguments = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(arguments) >= 8:
                    self.error(self.peek(), "Cannot have more than 8 arguments.")
                arguments.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break
        paren = self.consume(TokenType.RIGHT_PAREN, "Expect ')' after arguments.")
        return ast.Call(callee, paren, arguments)


    def primary(self):
        if self.match(TokenType.FALSE):
            return ast.Literal(False)
        elif self.match(TokenType.TRUE):
            return ast.Literal(True)
        elif self.match(TokenType.NIL):
            return ast.Literal(None)
        elif self.match(TokenType.NUMBER, TokenType.STRING):
            return ast.Literal(self.previous().literal)
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression.")
            return ast.Grouping(expr)
        elif self.match(TokenType.IDENTIFIER):
            return ast.Variable(self.previous())
        elif self.match(TokenType.THIS):
            return ast.This(self.previous())
        else:
            self.error(self.peek(), "Expect expression.")

    def consume(self, ttype, message):
        if self.check(ttype):
            return self.advance()
        else:
            self.error(self.peek(), message)

    def error(self, token, message):
        error(token, message)
        # TODO(plesslie): make parsing error exception to wrap token
        raise Exception("Failed to parse")

    def synchronize(self):
        # TEMP TEMP: if this gets called, I probably have a bug, so don't
        # silently swallow errors
        sys.stderr.write("WARNING SYNCHRONIZE CALLED\n")
        self.advance()
        while self.peek().ttype != TokenType.EOF:
            if self.previous().ttype == TokenType.SEMICOLON:
                return
            ttype = self.peek().ttype
            if ttype in (TokenType.CLASS, TokenType.FUN, TokenType.VAR,
                    TokenType.FOR, TokenType.IF, TokenType.WHILE,
                    TokenType.PRINT, TokenType.RETURN):
                break
            self.advance()

    def statement(self):
        if self.match(TokenType.PRINT):
            return self.print_statement()
        elif self.match(TokenType.LEFT_BRACE):
            return ast.Block(self.block())
        elif self.match(TokenType.IF):
            return self.if_statement()
        elif self.match(TokenType.WHILE):
            return self.while_statement()
        elif self.match(TokenType.FOR):
            return self.for_statement()
        elif self.match(TokenType.RETURN):
            return self.return_statement()
        else:
            return self.expression_statement()

    def return_statement(self):
        keyword = self.previous()
        if self.check(TokenType.SEMICOLON):
            value = None
        else:
            value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after return value.")
        return ast.Return(keyword, value)

    def for_statement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'for'.")
        if self.match(TokenType.SEMICOLON):
            initializer = None
        elif self.match(TokenType.VAR):
            initializer = self.var_declaration()
        else:
            initializer = self.expression_statement()
        if self.check(TokenType.SEMICOLON):
            condition = None
        else:
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after loop condition.")
        if self.check(TokenType.RIGHT_PAREN):
            # TODO: should really be called post-action or something like that
            increment = None
        else:
            increment = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after for clauses.")
        body = self.statement()
        if increment is not None:
            body = ast.Block([body, ast.Expression(increment)])
        if condition is None:
            condition = ast.Literal(True)
        body = ast.While(condition, body)
        if initializer is not None:
            body = ast.Block([initializer, body])
        return body


    def while_statement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'while'.")
        condition = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after condition.")
        body = self.statement()
        return ast.While(condition, body)

    def if_statement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'if'.")
        condition = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after if condition.")
        then_branch = self.statement()
        if self.match(TokenType.ELSE):
            else_branch = self.statement()
        else:
            else_branch = None
        return ast.If(condition, then_branch, else_branch)

    def is_end(self):
        return self.peek().ttype == TokenType.EOF

    def block(self):
        statements = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_end():
            statements.append(self.declaration())
        self.consume(TokenType.RIGHT_BRACE, "Expect '}' after block.")
        return statements

    def declaration(self):
        try:
            if self.match(TokenType.VAR):
                return self.var_declaration()
            elif self.match(TokenType.CLASS):
                return self.class_declaration()
            elif self.match(TokenType.FUN):
                return self.function("function")
            else:
                return self.statement()
        except Exception:
            # TEMP TEMP
            raise
            # self.synchronize()
            # return None

    def class_declaration(self):
        name = self.consume(TokenType.IDENTIFIER, "Expect class name.")
        if self.match(TokenType.LESS):
            self.consume(TokenType.IDENTIFIER, "Expect superclass name.")
            superclass = ast.Variable(self.previous())
        else:
            superclass = None
        self.consume(TokenType.LEFT_BRACE, "Expect '{' before class body.")
        methods = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_end():
            methods.append(self.function("method"))
        self.consume(TokenType.RIGHT_BRACE, "Expect '}' after class body.")
        return ast.Class(name, superclass, methods)

    def function(self, kind):
        name = self.consume(TokenType.IDENTIFIER, "Expect {} name.".format(
            kind))
        parameters = []
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after {} name".format(
            kind))
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(parameters) >= 8:
                    self.error(self.peek(), "Cannot have mre than 8 parameters.")
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expect parameter name."))
                if not self.match(TokenType.COMMA):
                    break
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after parameters.")
        self.consume(TokenType.LEFT_BRACE, "Expect '{{' before {} body.".format(
            kind))
        body = self.block()
        return ast.Function(name, parameters, body)

    def var_declaration(self):
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name.")
        value = None
        if self.match(TokenType.EQUAL):
            value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after variable declaration.")
        return ast.Var(name, value)

    def print_statement(self):
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after value.")
        return ast.Print(value)

    def expression_statement(self):
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after expression.")
        return ast.Expression(expr)

    def parse(self):
        statements = []
        while self.peek().ttype != TokenType.EOF:
            statements.append(self.declaration())
        return statements


def error(token, message):
    if token.ttype == TokenType.EOF:
        report(token.line, " at end", message)
    else:
        report(token.line, " at '{}'".format(token.lexeme), message)


def report(line, where, message):
    sys.stderr.write("[line {}] Error{}: {}\n".format(
        line, where, message))


def run_program(prog):
    with open(prog, 'r') as f:
        source = '\n'.join(f.readlines())
    run(source)


def run_prompt():
    while True:
        try:
            line = input('> ')
            run(line)
        except EOFError:
            break


def run(source):
    tokens = scan_tokens(source)
    parser = Parser(tokens)
    stmts = parser.parse()
    interp = Interpreter()
    resolver = Resolver(interp)
    resolver.resolve(stmts)
    interp.interpret(stmts)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: jlox.py [script]")
        sys.exit(0)
    elif len(sys.argv) == 2:
        run_program(sys.argv[1])
    else:
        run_prompt()
    print("Bye.")
