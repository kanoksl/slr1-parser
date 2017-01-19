FA_INITIAL_STATE = "00"

# Each line is a transition in this format:
# <current_state> <symbol_class> <next_state> [<is_lookahead>]
FA_TRANSITIONS = r"""
00  #       01
00  a-z     03
00  A-Z     03
00  _       03
00  "       05
00  /       14
00  +       17
00  -       21
00  0-9     22
00  .       27
01  other   01
01  newline 02  lookahead
03  a-z     03
03  A-Z     03
03  0-9     03
03  _       03
03  other   04  lookahead
05  "       08
05  \       07
05  other   06
06  \       07
06  "       13
06  other   06
07  other   06
08  "       09
08  other   13  lookahead
09  "       11
09  \       10
09  other   09
10  other   09
11  "       12
11  other   09
12  "       13
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
"""

# The format is: <final_state> <token_type>
FA_FINAL_STATES = """
02  comment
04  identifier
13  string
15  op_//
16  op_/
18  op_++
19  op_+=
20  op_+
29  float
"""

# -------------------------------------------------------------------------- #


def build_automaton():
    """
    Convert the automaton definition text to dictionaries.
    Use FA_TRANSITIONS and FA_FINAL_STATES defined above.
    :return: A tuple (initial state, transitions, lookahead transitions, final state tokens).
    """

    transition_map = dict()  # dict: (state, symbol) -> (state)
    lookahead_set = set()  # set of (state, symbol) tuples
    token_map = dict()  # dict: (state) -> (token)

    for line in FA_TRANSITIONS.splitlines():
        line = line.strip()
        if not line: continue  # ignore empty lines

        state, symbol, next_state, *lookahead = line.split()
        # if symbol.startswith('\\'): symbol = symbol[1:]
        transition_map[state, symbol] = next_state
        if lookahead:
            lookahead_set.add((state, symbol))

        if symbol == SC_LETTER_LOWER:  # add separate (state, 'e') transition
            transition_map[state, 'e'] = next_state
        elif symbol == SC_LETTER_UPPER:  # add separate (state, 'E') transition
            transition_map[state, 'E'] = next_state

    for line in FA_FINAL_STATES.splitlines():
        line = line.strip()
        if not line: continue  # ignore empty lines

        state, token = line.split()
        token_map[state] = token

    return FA_INITIAL_STATE, transition_map, lookahead_set, token_map

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


TK_INVALID = 'invalid'


def scan_one(automaton, buffer, pos):
    """
    Scan the buffer starting at specified position until one token is found
    or the end of buffer is reached.
    :return: The found lexeme, token, and the position for the next scan.
    """
    start, transitions, lookahead_trans, final_states = automaton
    state = start
    lexeme = []

    while pos < len(buffer) and buffer[pos] in SS_WHITESPACE:
        pos += 1  # ignore whitespaces

    while state not in final_states and pos <= len(buffer):
        if pos == len(buffer):
            sym = ''
            classes = [SC_OTHER]
        else:
            sym = buffer[pos]
            classes = symbol_classes(buffer[pos])
        for symcls in classes:
            if (state, symcls) in transitions:
                next_state = transitions[state, symcls]
                # print('transition from ({}, {}) to ({})'.format(state, symcls, next_state))
                if (state, symcls) not in lookahead_trans:
                    lexeme.append(buffer[pos])
                    pos += 1
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
            continue
        if token in {'identifier', 'string', 'float'}:
            if lexeme not in table_hash:
                symbol_table.append((token, lexeme))
                table_hash[lexeme] = len(table_hash)
            token_stream.append((token, table_hash[lexeme]))
        else:
            token_stream.append(token)
    return token_stream, symbol_table


# -------------------------------------------------------------------------- #

TEST_TEXT = '''x + y // 12.35  # comment line 1
x + 145.23e+15 "hello world"
/ // ++ += """string string string"""  # comment line 3
z + some_name // -2.42
'''

if __name__ == '__main__':
    tokens, table = scan_all(build_automaton(), TEST_TEXT)

    print()
    print('SYMBOL_TABLE:')
    print('   i | token type | values')
    print('—' * 40)
    for i, (tk, lx) in enumerate(table):
        print(' {:3d} | {} | {}'.format(i, tk.ljust(10), lx))
    print('—' * 40)

    print()
    print('TOKEN_STREAM:')
    print('format: < token, symbol_table_addr (value) >')
    for tk in tokens:
        if type(tk) == tuple:
            print('  < {}, {} ({}) >'.format(*tk, table[tk[1]][1]))
        else:
            print('  < {} >'.format(tk))
    print()
