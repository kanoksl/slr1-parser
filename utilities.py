""" Utility functions. Mainly for printing things to the console. """

SYM_PROD = '->'

COLOR_ENABLED = True

_COLORS = dict(K=30, R=31, G=32, Y=33, B=34, M=35, C=36, W=37)
_COLOR_FORMAT = '\x1b[{};{}m{}\x1b[0m'


__all__ = ['COLOR_ENABLED', 'colorize',
           'print_grammar', 'print_tree']


def colorize(text, color, bold=False):
    """ Format the text for printing in color to terminal.

        :param text: String (or any object).
        :param color: One of the 8 available colors (as single character string):
            K (black), R (red), G (green), Y (yellow), B (blue), M (magenta), C (cyan), 
            or W (white). Actual color might be different due to terminal's config.
        :param bold: Set to True (or 1) for bold font or bright color.

        :return: The original string with color code added.
    """
    if not COLOR_ENABLED:
        return text
    return text if not COLOR_ENABLED else _COLOR_FORMAT.format(_COLORS[color], int(bold), text)


def print_grammar(grammar, color=True):
    """ Print all information of the given grammar to the console. 
    
        :param grammar: An instance of Grammar class defined in parser.py.
        :param color: Whether to use colors for different types of symbols or not.
            (Some console might not support displaying colored texts.)
    """

    # Function for colorizing grammar symbols.
    _color = grammar.colorize if color else (lambda string: string)

    def _list(symbols, sep=', '):
        """ Convert a list (or set) of grammar symbols to colorized string for printing. """
        symbols = sorted(symbols) if type(symbols) not in {list, tuple} else symbols
        return sep.join([_color(sym) for sym in symbols])

    print('Displaying Context-Free Grammar...')
    print('- Start Symbol = {}'.format(_color(grammar.start_symbol)))
    print('- Non-terminals ({}) = {}'.format(len(grammar.nonterminals), _list(grammar.nonterminals)))
    print('- Terminals ({}) = {}'.format(len(grammar.terminals), _list(grammar.terminals)))

    print('\n' '- Production Rules:')
    counter = 1
    for nt in grammar.nonterminals:
        rule = grammar.rule_dict[nt][0]
        print('    {:2d}: {} {} {}'.format(counter, _color(nt), SYM_PROD, _list(rule, sep=' ')))
        counter += 1
        indent = ' ' * (len(nt) + len(SYM_PROD))
        for rule in grammar.rule_dict[nt][1:]:
            print('    {:2d}: {}| {}'.format(counter, indent, _list(rule, sep=' ')))
            counter += 1

    if grammar.firsts and grammar.follows:
        width = max(max([len(nt) for nt in grammar.nonterminals]), max([len(t) for t in grammar.terminals]))
        print('\n' '- FIRST sets of non-terminals:')
        for nt in grammar.nonterminals:
            print('    {} : {{ {} }}'.format(_color(str(nt).ljust(width)), _list(grammar.firsts[nt])))
        print('\n' '- FOLLOW sets of non-terminals:')
        for nt in grammar.nonterminals:
            print('    {} : {{ {} }}'.format(_color(str(nt).ljust(width)), _list(grammar.follows[nt])))
        print('\n' '- FOLLOW sets of terminals:')
        for t in grammar.terminals:
            print('    {} : {{ {} }}'.format(_color(str(t).ljust(width)), _list(grammar.follows[t])))
    else:
        print('First and follow sets have not been analyzed yet.')


def print_parsing_table(parser, color=True):
    """ Print the SLR(1) parsing table of the given parser to the console.
    
        :param parser: An instance of Parser class defined in parser.py.
        :param color: Whether to use colors for different types of symbols or not.
            (Some console might not support displaying colored texts.)
    """
    colwidth = 10
    terminals = sorted(parser.grammar.terminals)
    terminals.append('$')
    nonterminals = parser.grammar.nonterminals
    cw_t = [max(len(t), 4) for t in terminals]
    cw_nt = [max(len(nt), 4) for nt in nonterminals]
    header = ('State | ' + ' '.join([str(t).center(cw_t[j]) for j, t in enumerate(terminals)])
          + ' | ' + ' '.join([str(nt).center(cw_nt[j]) for j, nt in enumerate(nonterminals)]) + ' |')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(len(parser.states)):
        print('{:5d} | '.format(i), end='')
        for j, t in enumerate(terminals):
            print((parser.action.get((i, t), '-') + str(parser.goto.get((i, t), '-'))).center(cw_t[j]), end='')
            print(' ', end='')
        print('| ', end='')
        for j, nt in enumerate(nonterminals):
            print(str(parser.goto.get((i, nt), '-')).center(cw_nt[j]), end='')
            print(' ', end='')
        print('|')
    print('-' * len(header))


def print_tree(node, indent=0, root=True, paren=('[', ']')):
    """ Print the tree structure consisting of TreeNode objects recursively. 
    
        :param node: An instance of TreeNode class defined in parser.py.
        :param indent: Number of spaces for indentation of this node.
        :param root: Whether this node is the root or not.
        :param paren: A tuple of 2 strings to use in place of parentheses.
    """
    print(paren[0] + node.dstr, end='')
    if node.children:
        print(' ', end='')
        child_indent = indent + len(str(node.data)) + 2
        print_tree(node.children[0], child_indent, root=False, paren=paren)
        for child in node.children[1:]:
            print('\n' + (' ' * child_indent), end='')
            print_tree(child, child_indent, root=False, paren=paren)
    print(paren[1], end='')
    if root:  # Insert new line at the end.
        print()


