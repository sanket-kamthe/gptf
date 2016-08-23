# -*- encoding: utf-8 -*-
"""Provides utility functions."""
from builtins import object
from functools import wraps, partial
import re
import os

def flip(func):
    """Reverses the arguments of `func`.

    Args:
        func (Callable[[A0, A1, ..., AN], B]): A function.

    Returns:
        (Callable[[AN, AN-1, ..., A1, A0], B]): `func`, with the order of
        arguments reversed.

    """
    @wraps(func)
    def flipped(*args, **kwargs):
        args = list(args)
        args.reverse()
        return func(*args, **kwargs)
    return flipped

def isclassof(classinfo):
    """Creates a partial application of isinstance.

    Args:
        classinfo: The classinfo argument for isinstance.
    
    Returns:
        (Callable[[object], bool]): A function `f(obj)` that returns 
        `isinstance(object, classinfo)`
          
    """
    return partial(flip(isinstance), classinfo)

def isattrof(name):
    """Creates a partial application of hasattr.

    Args:
        name: The name of the attribute to check for.
    
    Returns:
        (Callable[[object], bool]): A function `f(obj)` that returns 
        `hasattr(object, name)`
          
    """
    return partial(flip(hasattr), name)

def construct_table(cols, headings, fmt):
    """Constructs a table from the given columns and rows for display.

    Args:
        cols (Iterable[Iterable[str]]): The columns of the table.
            The columns should be of the same length.
        headings (Iterable[str]): The headings for the columns.
            `len(headings)` should be the same as `len(cols)`.
        fmt ('fancy' | 'plain' | 'html'): The format for the table.
            `'fancy'` returns a table with fancy formatting using box
            drawing characters.
            `'plain'` returns a table using only ascii characters.
            `'html'` returns an html table.

    Returns:
        (str): A string containing a table.

    """
    columnlength = len(cols[0])
    for col in cols:
        assert len(col) == columnlength
    assert len(cols) == len(headings)

    def columnwidth(category, header):
        return max(printed_width(header), *map(printed_width, category))

    def printed_width(x):
        if fmt == "fancy":
            return len(strip_ansi_escape_codes(x))
        else:
            return len(x)

    col_ws = tuple(columnwidth(c, h) for c, h, in zip(cols, headings))

    def format_widths(row):
        return [w + len(x) - printed_width(x) for w, x in zip(col_ws, row)]

    prefix = []
    suffix = []
    if fmt == "fancy":
        tab_row = " │ ".join("{{:{:d}}}" for _ in range(len(cols)))
        head_sep = "═╪═".join("".join("═" for _ in range(w)) for w in col_ws)
    elif fmt == "plain":
        tab_row = " | ".join("{{:{:d}}}" for _ in range(len(cols)))
        head_sep = "-+-".join("".join("-" for _ in range(w)) for w in col_ws)
    elif fmt == "html":
        prefix = ["<table id='params' width=100%>"]
        suffix = ["</table>"]
        tab_row = "  <tr>" + os.linesep
        tab_row += os.linesep.join("    <td>{{:{:d}}}</td>"
                for _ in range(len(cols)))
        tab_row += + os.linesep + "  </tr>"
        head_sep = ""
    else:
        raise ValueError("fmt must be one of 'fancy', 'plain', 'html'")

    lines = [tab_row.format(*format_widths(headings)).format(*headings)]
    if head_sep: lines.append(head_sep)
    for row in zip(*cols):
        lines.append(tab_row.format(*format_widths(row)).format(*row))
    return os.linesep.join(prefix + lines + suffix)

def prefix_lines(prefix, string):
    """Prefixes each line in `string` with `prefix`."""
    lines = string.split(os.linesep)
    return prefix + "{}{}".format(os.linesep, prefix).join(lines)

ansi_escape_re = re.compile(r'\x1b[^m]*m')
def strip_ansi_escape_codes(string):
    return ansi_escape_re.sub('', string)
