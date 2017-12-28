#!/usr/bin/env python3

import sys
import readline # noqa


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


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: jlox.py [script]")
        sys.exit(0)
    elif len(sys.argv) == 2:
        run_program(sys.argv[0])
    else:
        run_prompt()
    print("Bye.")
