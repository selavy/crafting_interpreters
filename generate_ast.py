#!/usr/bin/env python3

import sys


def gentype(f, node):
    klass = node[0]
    members = node[1]

    f.write('class {klass}(object):\n'.format(klass=klass))
    args = ['self'] + members
    f.write('    def __init__({args}):\n'.format(
        args=', '.join(args)))

    for member in members:
        f.write('        self.{name} = {name}\n'.format(
            name=member))
    f.write('\n')
    f.write('    def __str__(self):\n')
    f.write('        return "{klass}"\n'.format(klass=klass))
    f.write('\n')
    f.write('    def accept(self, visitor):\n')
    f.write('        return visitor.visit_{klass}(self)\n'.format(
        klass=klass.lower()))
    f.write('\n')


if __name__ == '__main__':
    nodes = (
            # Expr
            ('Binary', ['left', 'operator', 'right']),
            ('Grouping', ['expression']),
            ('Literal', ['value']),
            ('Unary', ['operator', 'right']),
            ('Variable', ['name']),
            # Stmt
            ('Expression', ['expression']),
            ('Print', ['expression']),
            ('Var', ['name', 'initializer']),
            ('Assign', ['name', 'value']),
            ('Block', ['statements']),
            ('If', ['condition', 'then_branch', 'else_branch']),
            ('Logical', ['left', 'operator', 'right']),
            ('While', ['condition', 'body']),
            ('Call', ['callee', 'paren', 'arguments']),
            ('Function', ['name', 'parameters', 'body']),
            ('Return', ['keyword', 'value']),
            ('Class', ['name', 'methods']),
            ('Get', ['object', 'name']),
            ('Set', ['object', 'name', 'value']),
    )
    for node in nodes:
        gentype(sys.stdout, node)


