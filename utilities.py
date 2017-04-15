""" Utility functions. Mainly for printing things to the console. """

SYM_PROD = '->'  # Symbol used for production rules.
SYM_EMPTY = 'lambda'  # Representing empty string in grammar.
SYM_FINAL = '$'  # End-marker symbol.

__all__ = ['COLOR_ENABLED', 'colorize', 'SyntaxHighlighter'
           'print_grammar', 'print_parser_dfa', 'print_parsing_table', 'print_tree']

COLOR_ENABLED = True

_COLORS = dict(K=30, R=31, G=32, Y=33, B=34, M=35, C=36, W=37)
_COLOR_FORMAT = '\x1b[{};{}m{}\x1b[0m'


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
    return _COLOR_FORMAT.format(_COLORS[color], int(bold), text)


class SyntaxHighlighter:
    def __init__(self, grammar):
        self.grammar = grammar
        """ An instance of the Grammar class defined in parser.py. """
        self.colormap = dict()
        """ A dict for mapping grammar symbols to their colors. """

        self._build_colormap()

    def _build_colormap(self):
        color_terminals = 'G'
        color_nonterminals = 'B'
        color_specials = 'R'
        for x in self.grammar.terminals:
            self.colormap[x] = color_terminals
        for x in self.grammar.nonterminals:
            self.colormap[x] = color_nonterminals
        for x in [SYM_EMPTY, SYM_FINAL]:
            self.colormap[x] = color_specials

    def c(self, symbol):
        """ Get the color-formatted string of a grammar symbol. """
        if symbol in self.colormap:
            return colorize(symbol, self.colormap[symbol], bold=True)
        return symbol

    def p(self, symbol):
        """ Print the colorized symbol to the standard output. """
        print(self.c(symbol), end='')

    def l(self, symbols, sep=', '):
        """ Get the color-formatted string of a collection of grammar symbols.
            If the collection is not a list or tuple, the output list of symbols 
            will be sorted alphabetically. 
        """
        symbols = sorted(symbols) if type(symbols) not in {list, tuple} else symbols
        return sep.join([self.c(sym) for sym in symbols])


def print_grammar(grammar, color=True):
    """ Print all information of the given grammar to the console. 
    
        :param grammar: An instance of Grammar class defined in parser.py.
        :param color: Whether to use colors for different types of symbols or not.
            (Some console might not support displaying colored texts.)
    """

    h = SyntaxHighlighter(grammar)
    if not color:  # Disable SyntaxHighlighter by clearing its colormap.
        h.colormap.clear()

    print('Context-Free Grammar:')
    print('- Start Symbol = {}'.format(h.c(grammar.start_symbol)))
    print('- Terminals ({}) = {}'.format(len(grammar.terminals), h.l(grammar.terminals)))
    print('- Non-Terminals ({}) = {}'.format(len(grammar.nonterminals), h.l(grammar.nonterminals)))

    print('\n' '- Production Rules:')
    counter = 1
    for x in grammar.nonterminals:
        rule = grammar.rule_dict[x][0]
        print('    {:2d}: {} {} {}'.format(counter, h.c(x), SYM_PROD, h.l(rule, sep=' ')))
        counter += 1
        indent = ' ' * (len(x) + len(SYM_PROD))
        for rule in grammar.rule_dict[x][1:]:
            print('    {:2d}: {}| {}'.format(counter, indent, h.l(rule, sep=' ')))
            counter += 1

    if grammar.firsts and grammar.follows:
        # Max len of grammar symbol, plus len of color format string.
        width = max(max([len(x) for x in grammar.nonterminals]),
                    max([len(x) for x in grammar.terminals]))
        width += 11 if color else 0
        print('\n' '- FIRST sets of non-terminals:')
        for x in grammar.nonterminals:
            print('    {} : {{ {} }}'.format(h.c(x).ljust(width), h.l(grammar.firsts[x])))
        print('\n' '- FOLLOW sets of non-terminals:')
        for x in grammar.nonterminals:
            print('    {} : {{ {} }}'.format(h.c(x).ljust(width), h.l(grammar.follows[x])))
        print('\n' '- FOLLOW sets of terminals:')
        for x in grammar.terminals:
            print('    {} : {{ {} }}'.format(h.c(x).ljust(width), h.l(grammar.follows[x])))
    else:
        print('FIRST and FOLLOW sets have not been computed yet.')


def print_parser_dfa(parser):
    """ Print the states and the LR items in each state of the given SLR(1) parser. 
    
        :param parser: An instance of Parser class defined in parser.py.
    """
    print('\n' 'Number of parser DFA states =', len(parser.states))
    for i, st_items in enumerate(parser.states):
        print('\n' 'State {:d} LR items:'.format(i))
        for (idx, dp) in sorted(st_items):
            head, *body = list(parser.rule_list[idx])
            body.insert(dp, '.')
            print('  ({:d}, {:d}): {} {} {}'.format(idx, dp, head, SYM_PROD, ' '.join(body)))


def print_parsing_table(parser):
    """ Print the SLR(1) parsing table of the given parser to the console.
    
        :param parser: An instance of Parser class defined in parser.py.
    """
    terminals = sorted(parser.grammar.terminals)
    terminals.append('$')
    nonterminals = parser.grammar.nonterminals

    min_cw = 3  # Minimum column width.
    cw_t = [max(len(x), min_cw) for x in terminals]
    cw_n = [max(len(x), min_cw) for x in nonterminals]

    header = ('| State | {} | {} |'.format(
        ' '.join([str(x).center(cw_t[j]) for j, x in enumerate(terminals)]),
        ' '.join([str(x).center(cw_n[j]) for j, x in enumerate(nonterminals)])))
    h_line = ('+-------+{}+{}+'.format(
        '-' * (sum(cw_t) + len(cw_t) + 1),
        '-' * (sum(cw_n) + len(cw_n) + 1)))

    print(h_line)
    print(header)
    print(h_line)

    # ACTION and GOTO function for state _i and symbol _x.
    action = lambda _i, _x: str(parser.action.get((_i, _x), '-'))
    goto = lambda _i, _x: str(parser.goto.get((_i, _x), '-'))

    for i in range(len(parser.states)):
        data_t = ' '.join([(action(i, x) + goto(i, x)).center(cw_t[j]) for j, x in enumerate(terminals)])
        data_n = ' '.join([goto(i, x).center(cw_n[j]) for j, x in enumerate(nonterminals)])
        print('| {:5d} | {} | {} |'.format(i, data_t, data_n))

    print(h_line)


def print_tree(node, indent=0, root=True, paren=('[', ']')):
    """ Print the tree structure consisting of TreeNode objects recursively. 
    
        :param node: An instance of TreeNode class defined in parser.py.
        :param indent: Number of spaces for indentation of this node.
        :param root: Whether this node is the root or not.
        :param paren: A tuple of 2 strings to use in place of parentheses.
    """
    print(paren[0] + node.pr_str, end='')
    if node.children:
        print(' ', end='')
        child_indent = indent + node.pr_len + 2
        print_tree(node.children[0], child_indent, root=False, paren=paren)
        for child in node.children[1:]:
            print('\n' + (' ' * child_indent), end='')
            print_tree(child, child_indent, root=False, paren=paren)
    print(paren[1], end='')
    if root:  # Insert new line at the end.
        print()
