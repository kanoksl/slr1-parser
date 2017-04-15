#!/usr/bin/env python3

import utilities as util

# region:: Context-Free Grammar Class ::


SYM_PROD = '->'  # Symbol used for production rules.
SYM_EMPTY = 'lambda'  # Representing empty string in grammar.
SYM_FINAL = '$'  # End-marker symbol.


class Grammar:
    """ Class representing a context-free grammar. """

    def __init__(self, filepath):
        """ Create a new grammar with production rules from the given text file. """
        self.start_symbol = None
        """ Start symbol of the grammar. Must be a non-terminal symbol. """
        self.terminals = set()
        """ Set of all terminals (tokens) found in production rules. """
        self.nonterminals = list()
        """ List of all non-terminals, ordered as defined in production rules. """
        self.rule_dict = dict()
        """ Production rules. Key is a non-terminal symbol X. Value is a list of tuples
            [(A1, A2, ..., An), (B1, B2, ..., Bn), ...] representing rules X -> A1 A2 ... An
            | B1 B2 ... Bn | ... """
        self.rule_list = list()
        """ List of production rules ordered by added time (same order as in the grammar file
            if the load() function is used. Each list item is a tuple (X, A1, A2, ..., An), 
            representing a rule X -> A1 A2 ... An. """
        self.firsts = dict()
        """ Dict of first sets of each non-terminal symbols. """
        self.follows = dict()
        """ Dict of follow sets of each grammar symbols. """

        self._load(filepath)
        self._analyze()

    def _load(self, filepath):
        """ Load a grammar definition from a text file. Also list all non-terminals
            and set the start symbol automatically. """
        all_symbols = list()
        with open(filepath) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if not line or line.startswith(';'):
                    line = f.readline()
                    continue  # Ignore blank/comment lines.
                tokens = line.split()
                if tokens[1] != SYM_PROD:  # Must use this symbol to define a rule.
                    raise RuntimeError('Invalid grammar rule definition: ' + line)
                symbol, body = tokens[0], tokens[2:]
                all_symbols.extend(body)

                # Add the production rule rule and the left-hand-side non-terminal.
                if symbol not in self.rule_dict:
                    self.nonterminals.append(symbol)
                    self.rule_dict[symbol] = list()
                self.rule_dict[symbol].append(tuple(body))
                self.rule_list.append((symbol, *body))

                line = f.readline()

        # List all terminal symbols.
        for s in all_symbols:
            if s not in self.rule_dict and s != SYM_EMPTY:
                self.terminals.add(s)
        self.start_symbol = self.nonterminals[0]

    def _analyze(self):
        """ Find the first and follow sets of each grammar symbol. """
        self.firsts = find_firsts(self.rule_list, self.terminals, self.nonterminals)
        self.follows = find_follows(self.rule_list, self.terminals, self.nonterminals,
                                    self.start_symbol, self.firsts)

    def all_symbols(self):
        symbols = self.nonterminals.copy()
        symbols.extend(sorted(self.terminals))
        return symbols

    def colorize(self, symbol):
        """ Colorize the symbol according to its type for printing. """
        if symbol in {SYM_EMPTY, SYM_FINAL}:
            color, bold = 'R', True
        elif symbol in self.terminals:
            color, bold = 'G', True
        else:
            color, bold = 'B', True
        return util.colorize(symbol, color, bold)


def find_firsts(rule_list, terminals, nonterminals):
    """ Find the first sets of given grammar symbols from the given production rules.
    
        :param rule_list: List of production rule tuples (H, X1, ... Xn).
        :param terminals: Collection of terminal symbols in the grammar.
        :param nonterminals: Collection of non-terminal symbols in the grammar.
        
        :return: A dict in form of { grammar_symbol: its_first_set }.
    """
    first_sets = dict()
    first_sets[SYM_EMPTY] = {SYM_EMPTY}
    for t in terminals:
        first_sets[t] = {t}
    for nt in nonterminals:
        first_sets[nt] = set()

    updated = True
    while updated:
        updated = False
        for head, *body in rule_list:  # For each rule H -> X1 X2 ... Xn;
            for x in body:
                addition = first_sets[x] - {SYM_EMPTY}
                if not addition.issubset(first_sets[head]):
                    first_sets[head].update(addition)
                    updated = True
                if SYM_EMPTY not in first_sets[x]:
                    break
            else:  # Did not break; SYM_EMPTY is in all of Xi in the body.
                first_sets[head].add(SYM_EMPTY)

    return first_sets


def find_follows(rule_list, terminals, nonterminals, start_symbol, first_sets):
    """ Find the follow sets of given grammar symbols from the given production rules.
    
        :param rule_list: List of production rule tuples (H, X1, ... Xn).
        :param terminals: Collection of terminal symbols in the grammar.
        :param nonterminals: Collection of non-terminal symbols in the grammar.
        :param start_symbol: Start symbol of the grammar.
        :param first_sets: A dict of first sets of each grammar symbol.
        
        :return: A dict in form of { grammar_symbol: its_follow_set }.
    """
    _cache = dict()  # Cache the result of first().

    def first(rule_body):
        """ Find the first set of some rule body X1 ... Xn. Which is equal to First[X1]
            union First[Xi] for each subsequent Xi if lambda is in First[Xj] for all j < i.
            If all first sets of X1, ..., Xn contain lambda, then First[X1...Xn] contains
            lambda also.
            
            The result is cached in a dict for potential performance increase.
        """
        if not rule_body:
            return {SYM_EMPTY}
        rule_body = tuple(rule_body)
        if rule_body in _cache:
            return _cache[rule_body]

        firsts = set()
        for xi in rule_body:
            firsts.update(first_sets[xi] - {SYM_EMPTY})
            if SYM_EMPTY not in first_sets[xi]:
                break
        else:  # Did not break; lambda is in all of First[Xi].
            firsts.add(SYM_EMPTY)

        _cache[rule_body] = firsts
        return firsts

    # Initialize with empty sets.
    follow_sets = {nt: set() for nt in nonterminals}
    for t in terminals:
        follow_sets[t] = set()

    # For each rule H -> X1 X2 ... Xn, we have Follow[Xi] = First[X(i+1)...Xn] - {lambda}.
    for _, *body in rule_list:
        for i, x in enumerate(body):
            if x == SYM_EMPTY: continue
            follow_sets[x].update(first(body[i + 1:]) - {SYM_EMPTY})

    # Follow[StartSymbol] must have $.
    follow_sets[start_symbol].add(SYM_FINAL)

    updated = True
    while updated:
        updated = False
        for head, *body in rule_list:  # For each rule H -> X1 X2 ... Xn;
            for i, x in enumerate(body):  # For each Xi in the body of the rule;
                if x == SYM_EMPTY: continue
                if SYM_EMPTY not in first(body[i + 1:]): continue
                if follow_sets[head].issubset(follow_sets[x]): continue
                # Add Follow[Xi] to Follow[H] if lambda is in First[X(i+1)...Xn].
                updated = True
                follow_sets[x].update(follow_sets[head])

    return follow_sets


# endregion ------------------------------------------------------------------------------ ::
# region:: SLR(1) Parser Implementation ::


# Parser actions.
ACT_SHIFT = 's'
ACT_REDUCE = 'r'
ACT_ACCEPT = 'ac'


class Parser:
    """ SLR(1) bottom-up parser. """

    def __init__(self, grammar):
        """ Initialize a parser for the given context-free grammar. """
        self.grammar = grammar
        """ Context-free grammar for this parser. """
        self.table = dict()
        """ The SLR parsing table. The key is a tuple (state, symbol), and the value
            is a tuple (action, next_state). """

        self.states = list()
        self.action = dict()
        """ The parsing action table. The key is a tuple (state, terminal), and the
            value can be either Shift, Reduce, or Accept. If a pair (state, terminal)
            is not in this dict's keys, then parsing error occurs."""
        self.goto = dict()
        """ The transition table. """

        self._build_table()

    def _build_table(self):
        start_symbol_old = self.grammar.start_symbol
        rule_list = self.grammar.rule_list.copy()
        # Create augmented grammar by adding new start symbol.
        start_symbol = str(start_symbol_old) + "'"
        rule_list.insert(0, (start_symbol, start_symbol_old))

        # Note that an LR item can be represented by a tuple (rule_idx, dot_pos).
        def closure(items):
            """ Return a set of LR items that is the closure of the given ones. """
            clsr = items.copy() if type(items) == set else {items}
            checklist = list(clsr)
            updated = True
            while updated and checklist:
                updated = False
                rule_idx, dot_pos = checklist.pop()
                if dot_pos + 1 > len(rule_list[rule_idx]) - 1:
                    continue
                right_of_dot = rule_list[rule_idx][dot_pos + 1]  # The symbol next to the dot.
                if right_of_dot in self.grammar.terminals:
                    clsr.add((rule_idx, dot_pos))
                    updated = True
                    continue
                # if right_of_dot == SYM_EMPTY: continue
                rules = self.grammar.rule_dict[right_of_dot]
                for prod in rules:
                    prod_idx = rule_list.index((right_of_dot, *prod))
                    new_item = (prod_idx, 0)
                    if new_item not in clsr:
                        clsr.add(new_item)
                        checklist.append(new_item)
                        updated = True
            return clsr

        def print_items(items):
            """ Print a collection of LR-item tuples in easily-readable format. """
            items = items if type(items) in {set, list} else [items]
            for (rule_idx, dot_pos) in sorted(items):
                rule = rule_list[rule_idx]
                body = list(rule[1:])
                body.insert(dot_pos, '.')
                print('  {} {} {}'.format(rule[0], SYM_PROD, ' '.join(body)))

        print('LR Items:')
        print_items(closure({(0, 0)}))

        def goto(items, symbol):
            """ If A -> a . X b is in items, then A -> a X . b is in goto(items, X) """
            gotoset = set()
            for ruleidx, dotpos in items:
                if dotpos + 1 > len(rule_list[ruleidx]) - 1:
                    continue
                if rule_list[ruleidx][dotpos + 1] == symbol:
                    gotoset.update(closure({(ruleidx, dotpos + 1)}))
            return gotoset

        def item_collection():
            collection = [closure({(0, 0)})]
            updated = True
            while updated:
                updated = False
                for j, items in enumerate(collection):
                    for x in self.grammar.all_symbols():
                        gt = goto(items, x)
                        if gt and gt not in collection:
                            collection.append(gt)
                            if x in self.grammar.terminals:
                                self.action[j, x] = ACT_SHIFT
                            self.goto[j, x] = len(collection) - 1
                            updated = True
                        elif gt in collection:
                            if x in self.grammar.terminals:
                                self.action[j, x] = ACT_SHIFT
                            self.goto[j, x] = collection.index(gt)
            return collection

        col = item_collection()
        print('\nNumber of states =', len(col))
        for i, items in enumerate(col):
            print('State', i, 'LR items:')
            print_items(items)

        for j, items in enumerate(col):
            if (0, 1) in items:
                self.action[j, SYM_FINAL] = ACT_ACCEPT
            else:
                complete_items = [ruleidx for ruleidx, dps in items if dps == len(rule_list[ruleidx]) - 1]
                for r in complete_items:
                    head = rule_list[r][0]
                    for x in self.grammar.follows[head]:
                        if self.action.get((j, x), None) != ACT_SHIFT:
                            self.action[j, x] = ACT_REDUCE
                            self.goto[j, x] = r  # Index of rule to use for reduction.

        self.states = col
        util.print_parsing_table(self)

    def parse(self, tokens):
        """ Build a parse tree from the given stream of tokens and the parsing table.
            :param tokens: An iterator that yields token in left-to-right order.
        """
        print('Begin parsing...')
        current_state = 0
        stack = [(TreeNode(data=SYM_FINAL), current_state)]
        tk = next(tokens)
        while True:
            if (current_state, tk) not in self.action:
                print('ERROR: unexpected token ' + self.grammar.colorize(tk) + ' at state ' + str(current_state))
                raise RuntimeError('Parsing error, unexpected token {} at state {}'.format(tk, current_state))
            elif self.action[current_state, tk] == ACT_SHIFT:
                print(' - action: shift ' + self.grammar.colorize(tk))
                current_state = self.goto[current_state, tk]
                stack.append((TreeNode(data=tk, dstr=self.grammar.colorize(tk)), current_state))
                tk = next(tokens)
            elif self.action[current_state, tk] == ACT_REDUCE:
                rule = self.grammar.rule_list[self.goto[current_state, tk] - 1]
                print(' - action: reduce [{} -> {}]'.format(self.grammar.colorize(rule[0]), 
                      ' '.join([self.grammar.colorize(sym) for sym in rule[1:]])))
                newnode = TreeNode(data=rule[0], dstr=self.grammar.colorize(rule[0]))
                for _ in range(len(rule) - 1):
                    node, current_state = stack.pop()
                    newnode.insert_child(node)
                current_state = stack[-1][1]
                current_state = self.goto[current_state, rule[0]]
                stack.append((newnode, current_state))
            elif self.action[current_state, tk] == ACT_ACCEPT:
                print(' - action: ' + util.colorize('ACCEPT', 'G'))
                node, _ = stack.pop()
                return node


# endregion ------------------------------------------------------------------------------ ::
# region:: Data Structure for Parse Tree ::


class TreeNode:
    """ A generic node for tree data structure. """

    def __init__(self, data, dstr=None, children=None):
        self.data = data
        """ Data object for this node. """
        self.dstr = dstr or str(data)
        """ String of the data for printing. """
        self.children = children or []
        """ List of child TreeNode objects ordered left-to-right. """

    def add_child(self, node):
        """ Add a children node to the right of this node's children list. """
        assert type(node) == TreeNode
        self.children.append(node)

    def insert_child(self, node, idx=0):
        """ Insert a children into this node's children list at the given index. """
        assert type(node) == TreeNode
        self.children.insert(idx, node)


# endregion ------------------------------------------------------------------------------ ::
# region:: Testing Code ::


def get_sample_tokens():
    # sample = 'repeat id = n + n * n until id < n'.split()
    sample = 'if ( n * n ) + n / n > id + ( id - id ) then read id else id = n < n'.split()
    # sample = 'id * id + id'.split()
    # sample = 'if if other else other'.split()
    sample.append(SYM_FINAL)
    return iter(sample)


def main(inputstr=None):
    print('// begin parser test...' '\n')

    input_tks = inputstr or 'if ( n * n ) + n / n > id + ( id - id ) then read id else id = n < n'
    tks_iter = input_tks.split()
    tks_iter.append(SYM_FINAL)
    tks_iter = iter(tks_iter)

    # Load the grammar from file.
    grammar = Grammar(filepath='def_grammar.txt')
    util.print_grammar(grammar, color=True)

    # Create a parser for the grammar.
    parser = Parser(grammar=grammar)

    # Parse sample tokens and print the tree.
    root_node = parser.parse(tokens=tks_iter)
    print('\n' 'Input tokens:')
    print(input_tks)
    print('\n' 'Parse tree for the input tokens:')
    util.print_tree(root_node)

    print('\n' '// test finished')
    return 0


def main_interactive(grammarfile='def_grammar.txt'):
    # Load the grammar from file.
    grammar = Grammar(filepath=grammarfile)
    util.print_grammar(grammar, color=True)

    # Create a parser for the grammar.
    parser = Parser(grammar=grammar)

    while True:
        input_tks = input('Enter a string of tokens to parse:\n >>> ')
        tks_iter = input_tks.split()
        tks_iter.append(SYM_FINAL)
        print('\n' 'Tokens: ' + ' '.join([grammar.colorize(t) for t in tks_iter]) + '\n')
        tks_iter = iter(tks_iter)
        try:
            root_node = parser.parse(tokens=tks_iter)
        except RuntimeError:
            root_node = TreeNode(data='PARSE ERROR')
        print('\n' 'Parse tree for the input tokens:')
        util.print_tree(root_node)
        print()

# endregion


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    # args = ' '.join(args) if len(args) > 1 else args[0]
    # main(args)
    if args:
        main_interactive(args[0])
    else:
        main_interactive()
