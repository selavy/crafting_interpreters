#!/usr/bin/env python3

import sys


def gentype(f, node):
    klass = node[0]
    members = node[1]

    f.write('class {klass}(object):\n'.format(klass=klass))
    args = ['self'] + members
    f.write('    def __init__({args}):\n'.format(
        args=', '.join(members)))

    for member in members:
        f.write('        self.{name} = {name}\n'.format(
            name=member))
    f.write('\n')
    f.write('    def __str__(self):\n')
    f.write('        return "{klass}"\n'.format(klass=klass))
    f.write('\n')
    f.write('    def visit(self, visitor):\n')
    f.write('        return visitor.visit_{klass}(self)\n'.format(
        klass=klass.lower()))
    f.write('\n')


if __name__ == '__main__':
    nodes = (
            ('Binary', ['left', 'operator', 'right']),
            ('Grouping', ['expression']),
            ('Literal', ['value']),
            ('Unary', ['operator', 'right']),
    )
    for node in nodes:
        gentype(sys.stdout, node)


