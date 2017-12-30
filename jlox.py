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
    START = auto()

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
        return '{ttype!s} {lexeme!s} {literal!s}'.format(
                ttype=self.ttype, lexeme=self.lexeme, literal=self.literal)

    def __repr__(self):
        return self.__str__()


class ASTPrinter(object):
    def __init__(self):
        pass

    def print_(self, ast):
        return ast.accept(self)

    def parenthesize(self, name, *args):
        results = [name]
        for expr in args:
            results.append(expr.accept(self))
        return "({})".format(' '.join(results))

    def visit_binary(self, expr):
        return self.parenthesize(expr.operator.lexeme, expr.left, expr.right)

    def visit_grouping(self, expr):
        return self.parenthesize("group", expr.expression)

    def visit_literal(self, expr):
        if expr.value is None:
            return "nil"
        else:
            return str(expr.value)

    def visit_unary(self, expr):
        return self.parenthesize(expr.operator.lexeme, expr.right)


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


def error(line, message):
    report(line, '', message)


def report(line, where, message):
    sys.stdout.stderr("[line {}] Error{}: {}\n".format(
        line, where, message))


def run_program(prog):
    # TODO: implement
    pass


def run_prompt():
    while True:
        try:
            line = input('> ')
            run(line)
        except EOFError:
            break


def run(line):
    print("input: '{}'".format(line))
    tokens = scan_tokens(line)
    import pprint
    pprint.pprint(tokens)


if __name__ == '__main__':
    lit1 = ast.Literal(value=1.)
    lit2 = ast.Literal(value=2.)
    op = Token(ttype=TokenType.PLUS, lexeme='+', literal=None, line=1)

    ast = ast.Binary(left=lit1, operator=op, right=lit2)
    print(ASTPrinter().print_(ast))

    # if len(sys.argv) > 2:
    #     print("Usage: jlox.py [script]")
    #     sys.exit(0)
    # elif len(sys.argv) == 2:
    #     run_program(sys.argv[0])
    # else:
    #     run_prompt()
    # print("Bye.")
