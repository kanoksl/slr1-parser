FA_INITIAL_STATE = '00'

# Each line is a transition in this format:
#   <current_state> <symbol_class> <next_state> [<is_lookahead>]
# Put '#' in front to comment out a line.
FA_TRANSITIONS = r'''
00  #       01
00  a-z     03
00  A-Z     03
00  _       03
00  "       05  no_lexeme
00  /       14
00  +       17
00  -       21
00  0-9     22
00  .       27
00  =       A01
A01 other   A02 lookahead
01  other   01
01  newline 02  lookahead
03  a-z     03
03  A-Z     03
03  0-9     03
03  _       03
03  other   04  lookahead
05  "       08  no_lexeme
05  \       07
05  newline INVALID
05  other   06
06  \       07  no_lexeme
06  "       13  no_lexeme
06  newline INVALID
06  other   06
07  newline INVALID
07  other   06
08  "       09  no_lexeme
08  other   13  lookahead
09  "       11
09  \       10
09  other   09
10  other   09
11  "       12
11  other   09
12  "       13  remove_3
12  other   09
14  /       15
14  other   16  lookahead
17  +       18
17  =       19
17  0-9     22
17  .       27
17  other   20  lookahead
21  0-9     22
21  .       27
22  0-9     22
22  .       23
22  e       24
22  E       24
23  0-9     23
23  e       24
23  E       24
23  other   29  lookahead
24  +       25
24  -       25
24  0-9     26
25  0-9     26
26  0-9     26
26  other   29  lookahead
27  0-9     28
28  0-9     28
28  e       24
28  E       24
28  other   29  lookahead
'''

# The format is: <final_state> <token_type> [<notes/comment>]
FA_FINAL_STATES = r'''
02  comment         # ...
04  identifier
13  string          "..." or """..."""
15  op_floor_divide //
16  op_divide       /
18  op_increment    ++
19  op_add_assign   +=
20  op_add          +
29  float
A02 op_equal
'''


# -------------------------------------------------------------------------- #


OPT_LOOKAHEAD = 'lookahead'  # mark a transition as lookahead
OPT_NO_LEXEME = 'no_lexeme'  # don't add char to the lexeme for transitions with this option
OPT_REMOVE_3 = 'remove_3'  # remove the last 3 chars from the lexeme


def build_automaton():
    """
    Convert the automaton definition text to dictionaries.
    Use FA_TRANSITIONS and FA_FINAL_STATES defined above.
    :return: A tuple (initial state, transitions, lookahead transitions,
                      transition special options, final state tokens).
    """

    transition_map = dict()  # dict: (state, symbol) -> (state)
    lookahead_set = set()  # set of (state, symbol) tuples
    options_dict = dict()  # dict: (state, symbol) -> (special option)
    token_map = dict()  # dict: (state) -> (token)

    for line in FA_TRANSITIONS.splitlines():
        line = line.strip()
        if not line or line[0] == '#':
            continue  # ignore empty lines or comments

        state, symbol, next_state, *options = line.split()
        transition_map[state, symbol] = next_state
        if OPT_LOOKAHEAD in options:
            lookahead_set.add((state, symbol))

        if OPT_NO_LEXEME in options:
            options_dict[state, symbol] = OPT_NO_LEXEME
        elif OPT_REMOVE_3 in options:
            options_dict[state, symbol] = OPT_REMOVE_3

    for line in FA_FINAL_STATES.splitlines():
        line = line.strip()
        if not line or line[0] == '#':
            continue  # ignore empty lines or comments

        state, token, *_ = line.split()
        token_map[state] = token

    return FA_INITIAL_STATE, transition_map, lookahead_set, options_dict, token_map


# -------------------------------------------------------------------------- #


# Symbol sets. Note: 'e' is not included in the letter sets.
SS_LETTER_LOWER = set('abcdefghijklmnopqrstuvwxyz')
SS_LETTER_UPPER = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
SS_LETTER = SS_LETTER_LOWER.union(SS_LETTER_UPPER)
SS_DIGIT = set('0123456789')
SS_WHITESPACE = set(' \t\n')

# Symbol classes
SC_LETTER_LOWER = 'a-z'
SC_LETTER_UPPER = 'A-Z'
SC_LETTER = 'a-z|A-Z'
SC_DIGIT = '0-9'
SC_WHITESPACE = 'whitespace'
SC_NEWLINE = 'newline'
SC_OTHER = 'other'


def symbol_classes(symbol):
    """
    Get the classes of a symbol, in order from most specific to most general.
    Example: 'c' is in classes 'c', lowercase letters, letters, and all symbols (other).
    :return: A list of symbol's classes.
    """
    classes = [symbol]
    if symbol == '\n':
        classes = [SC_NEWLINE]
    elif symbol in SS_DIGIT:
        classes.append(SC_DIGIT)
    elif symbol in SS_LETTER_LOWER:
        classes.extend((SC_LETTER_LOWER, SC_LETTER))
    elif symbol in SS_LETTER_UPPER:
        classes.extend((SC_LETTER_UPPER, SC_LETTER))
    elif symbol in SS_WHITESPACE:
        classes.append(SC_WHITESPACE)
    classes.append(SC_OTHER)
    return classes


# -------------------------------------------------------------------------- #


TK_INVALID = 'INVALID_TOKEN'

SZ_FLOAT = 4  # size of floats (32 bit)
SZ_CHAR = 1  # size of a single character in a string (8 bit ASCII)
SZ_MEMADDR = 4  # size of memory address (32 bit)


def scan_one(automaton, buffer, pos):
    """
    Scan the buffer starting at specified position until one token is found
    or the end of buffer is reached.
    :return: The found lexeme, token, and the position for the next scan.
    """
    start, transitions, lookahead_trans, tran_options, final_states = automaton
    state = start
    lexeme = []

    while pos < len(buffer) and buffer[pos] in SS_WHITESPACE:
        pos += 1  # ignore whitespaces

    while state not in final_states and pos <= len(buffer):
        if pos == len(buffer):
            char = ''
            classes = [SC_OTHER]
        else:
            char = buffer[pos]
            classes = symbol_classes(char)
        for char_cls in classes:
            if (state, char_cls) in transitions:
                next_state = transitions[state, char_cls]
                # print('transition from ({}, {}) to ({})'.format(state, char_cls, next_state))
                if (state, char_cls) not in lookahead_trans:
                    pos += 1
                    if tran_options.get((state, char_cls), None) != OPT_NO_LEXEME:
                        lexeme.append(char)
                    if tran_options.get((state, char_cls), None) == OPT_REMOVE_3:
                        lexeme = lexeme[:-3]
                state = next_state
                break
        else:  # did not break; no transition found
            # print('invalid symbol encountered at position: {:,d} : '.format(pos, buffer[pos]))
            pos += 1
            break

    lexeme = ''.join(lexeme)
    if state in final_states:
        token = final_states[state]
    else:
        token = TK_INVALID
    # print('scan_one returning lexeme: {} (token: {})'.format(lexeme, token))
    return lexeme, token, pos


def scan_all(automaton, buffer):
    """
    Scan the whole buffer with the given automaton.
    :return: A list of tokens found and a symbol table.
    """
    pos = 0
    token_stream = []
    symbol_table = []
    table_hash = dict()
    while pos < len(buffer):
        lexeme, token, pos = scan_one(automaton, buffer, pos)
        if token is TK_INVALID:
            if lexeme:
                token_stream.append(token + ': ' + lexeme)
            continue
        if token in {'identifier', 'string', 'float'}:
            if lexeme not in table_hash:
                size = SZ_MEMADDR
                if token == 'string':
                    size = (len(lexeme) + 1) * SZ_CHAR
                elif token == 'float':
                    size = SZ_FLOAT
                symbol_table.append((token, size, lexeme))
                table_hash[lexeme] = len(table_hash)
            token_stream.append((token, table_hash[lexeme]))
        else:
            token_stream.append(token)
    return token_stream, symbol_table


# -------------------------------------------------------------------------- #

PRINT_SEPARATOR = '—' * 79


def print_table(table):
    dg = 4  # number of digits in 'i' column
    wdt = 12  # width of 'Type' column; at least 4
    wda = 8  # width of 'Address' column.
    wds = 4  # width of 'Size' column; at least 4
    header = (' {:' + str(dg - 1) + 's}i | {:' + str(wdt) + 's} | {:'
              + str(wds) + 's} | Value').format('', 'Type', 'Size'.rjust(wds))
    row_format = ' {:' + str(dg) + 'd} | {:' + str(wdt) + 's} | {:' + str(wds) + 's} | {}'
    row_format_extra = (' {:' + str(dg) + 's} | {:' + str(wdt) + 's} | {:'
                        + str(wds) + 's} | ').format('', '', '') + '{}'

    print('Sizes are in bytes. Assuming size of float = {}, char = {}.'
          .format(SZ_FLOAT, SZ_CHAR))
    print(PRINT_SEPARATOR)
    print(header)
    print(PRINT_SEPARATOR)
    for i, (tk, sz, lx) in enumerate(table):
        size = (str(sz) if sz is not None else '-').rjust(wds)
        if '\n' not in lx:  # single-line entries
            print(row_format.format(i, tk, size, lx))
        else:  # multi-line string entries
            lxs = lx.splitlines()
            print(row_format.format(i, tk, size, lxs[0]))
            for lxj in lxs[1:]:
                print(row_format_extra.format(lxj))
    print(PRINT_SEPARATOR)


def print_stream(stream, table):
    print('Format: <token, i (value)>, where i = address in the symbol table.')
    print(PRINT_SEPARATOR)
    for tk in stream:
        if type(tk) == tuple:  # those with symbol-table entry
            val = table[tk[1]][2]
            print('  <{}, {} ({})>'.format(*tk, val))
        else:
            print('  <{}>'.format(tk))
    print('Total: {} {}'.format(len(stream), 'token' if len(stream) == 1 else 'tokens'))

# -------------------------------------------------------------------------- #


TEST_TEXT = '''
x = -100.00
x = x + y // 12.35  # comment 1
y = -4.2e12 / x++

# strings
s1 = "hello world"
s2 = """this string is triple-quoted;
it can be multi-line;
and it can contain " ""."""  # comment 2
s3 = "quote: \\" <- quote"
s1 += s2 + s3

f++ / g // h

# some floats here:
.123
0.23e+11
10e9
-7.001931E-13
'''

if __name__ == '__main__':
    tokens, symtable = scan_all(build_automaton(), TEST_TEXT)
    print('\nINPUT_BUFFER:')
    print(TEST_TEXT)
    print('\nSYMBOL_TABLE:')
    print_table(symtable)
    print('\nTOKEN_STREAM:')
    print_stream(tokens, symtable)
