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


class Interpreter(object):
    def __init__(self):
        self.had_error = False
        self.environment = Environment()

    def interpret(self, statements):
        self.had_error = False
        try:
            for stmt in statements:
                self.execute(stmt)
        except Exception as e:
            lox_runtime_error(e)

    def execute(self, stmt):
        stmt.accept(self)

    def visit_expression(self, stmt):
        self.evaluate(stmt.expression)
        return None

    def visit_print(self, stmt):
        value = self.evaluate(stmt.expression)
        print(str(value))
        return None

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
        return None

    def visit_variable(self, expr):
        return self.environment.get(expr.name)

    def visit_assign(self, expr):
        value = self.evaluate(expr.value)
        self.environment.assign(expr.name, value)
        return value


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


# REVISIT(plesslie): does this really to be its own class?
class Environment(object):
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def define(self, name, value):
        self.values[name] = value

    def get(self, name):
        try:
            return self.values[name.lexeme]
        except KeyError:
            if self.enclosing is not None:
                return self.enclosing.get(name)
            else:
                raise RuntimeError("Undefined variable '{}'.".format(
                    name.lexeme))

    def assign(self, name, value):
        try:
            self.values[name.lexeme] = value
        except KeyError:
            if self.enclosing is not None:
                self.enclosing.assign(name, value)
            else:
                raise RuntimeError("Undefined variable '{}'.".format(
                    name.lexeme))


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
        expr = self.equality()
        if self.match(TokenType.EQUAL):
            equals = self.previous()
            value = self.assignment()
            if isinstance(expr, ast.Variable):
                name = expr.name
                return ast.Assign(name, value)
            else:
                self.error(equals, "Invalid assignment target.")
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
            right = addition()
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
            return self.primary()

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
        else:
            return self.expression_statement()

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
            else:
                return self.statement()
        except Exception:
            self.synchronize()
            return None

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
