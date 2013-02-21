#TODO
# * simplify nested numbers of equivalent operations 2/(x*2)
# * Simplify same types when nested
from copy import deepcopy
import math
import re
from Gnuplot import Gnuplot

class InvalidFormatError(Exception):
    def __init__(self, line, reason):
        self.line = line
        self.reason = reason
    def __str__(self):
        return repr(self.line)+': '+repr(self.reason)
    def __repr__(self):
        return repr(self.line)+': '+repr(self.reason)

class InvalidCallError(Exception):
    def __init__(self, call, reason):
        self.call = call
        self.reason = reason
    def __str__(self):
        return repr(self.call)+': '+repr(self.reason)
    def __repr__(self):
        return repr(self.call)+': '+repr(self.reason)

class InvalidOperationError(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        return repr(self.reason)
    def __repr__(self):
        return repr(self.reason)

nums = '0123456789.'
alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
ops = '+-*/^'
seps = '=,'
oplist = ['+', '-', '*', '/', '^', '**']
opprec = {}

def split(l, sep):
    """Splits a list into a list of lists by a separating string sep."""
    res = [[]]
    for el in l:
        if el == sep:
            res.append([])
        else:
            res[-1].append(el)
    return res

def deepjoin(l):
    """Joins a list to a string, nested lists are enclosed by parentheses and their contents joined as well."""
    res = ''
    for el in l:
        if type(el) == list:
            res += '('+(deepjoin(el))+')'
        else:
            res += str(el)
    return res

def isvar(tok):
    """Returns True if tok is a valid variable name. Not allowed to start with a number, otherwise characters in alpha and nums are alloewd."""
    if type(tok) != str:
        return False
    if not tok[0] in alpha:
        return False
    for c in tok:
        if not c in alpha+nums:
            return False
    return True

def isnum(tok):
    """Returns True if tok is a number. Uses the Python float parser."""
    try:
        float(tok)
        return True
    except:
        return False

def isop(tok):
    """Returns True if tok is an operator as specified by oplist."""
    return tok in oplist

def issep(tok):
    for c in tok:
        if c not in seps:
            return False
    return True

def isass(tok):
    """Returns True if tok is the equality sign (assignment operator)."""
    return tok == '='

def islist(tok):
    """Returns True if tok is a list."""
    return type(tok) == list

def isprecop(term, opclass):
    """Determines operator precedence between term and op. True if term < op."""
    return isinstance(term, opprec[opclass])

def splitlists(l, f):
    """Evaluates all elements in l with f and returns two lists, one with the successes, the other with the failures."""
    l1 = []
    l2 = []
    for el in l:
        if f(el):
            l1.append(el)
        else:
            l2.append(el)
    return l1, l2

def splitnums(l):
    """Splits a list into two, one containing all floats from that list and the other containing the rest."""
    return splitlists(l, lambda el: isinstance(el, Number))

def flatten(l, full=True, level=0):
    """Removes nesting in lists, sets and tuples, splices nested containers into the same position of the outer list.
        full: specifies if all types of containers, even those not matching the outer, should be flattened.
        level: specifies how deep the flattening process should go. Further nested list will not be flattened."""
    if type(l) not in (list, tuple, set):
        return l
    res = [None for el in l]
    offset = 0
    level -= 1
    for i,el in enumerate(l):
        i = i+offset
        if full and type(el) in (list, tuple, set) or type(el) == type(l):
            if level != 0:
                splice = flatten(el, full, level)
            else:
                splice = el
            res[i:i+1] = splice
            offset += len(splice)-1
        else:
            res[i:i+1] = [el]
    if type(l) == tuple:
        return tuple(res)
    if type(l) == set:
        return set(res)
    return res

def makeargs(args=(), sort=False):
    """Returns a tuple of sorted arguments, removing duplicates."""
    args = flatten(args)
    if sort:
        return tuple(sorted(set(args)))
    else:
        seen = set()
        return tuple([arg for arg in args if str(arg) not in seen and not seen.add(str(arg))])

def matchvars(l1, l2):
    """Returns a union of the sets of variable lists provided."""
    return type(l1)(set(l1).intersection(set(l2)))

class Term():
    def eval(self, vars, funcs):
        return self
    def contains(self, name):
        return False
    def unapply(self, funcs):
        return self
    def apply(self, name, term):
        return self

class Number(Term):
    def __init__(self, number):
        self.number = float(number)
    def __repr__(self):
        return 'Number('+repr(self.number)+')'
    def __str__(self):
        return str(self.number)
    def __float__(self):
        return self.number

class Sum(Term):
    def __init__(self, summands):
        self.summands = summands
    def __repr__(self):
        return 'Sum('+', '.join([repr(summand) for summand in self.summands])+')'
    def __str__(self):
        return '+'.join([str(summand) for summand in self.summands])
    def eval(self, vars, funcs):
        nums, summands = splitnums([summand.eval(vars, funcs) for summand in self.summands])
        summands = reduce(lambda agg, term: agg+term.summands if isinstance(term, Sum) else agg+[term], summands, [])
        if len(nums) == 0:
            return Sum(summands)
        num = math.fsum(nums)
        if len(summands) == 0:
            return Number(num)
        if num != 0:
            summands.append(Number(num))
        if len(summands) == 1:
                return summands[0]
        return Sum(summands)
    def contains(self, name):
        #return reduce(lambda x, y: x or y.contains(name), self.summands, False)
        for summand in self.summands:
            if summand.contains(name):
                return True
        return False
    def unapply(self, funcs):
        self.summands = [summand.unapply(funcs) for summand in self.summands]
        return self
    def apply(self, name, term):
        self.summands = [summand.apply(name, term) for summand in self.summands]
        return self

class Difference(Term):
    def __init__(self, minuend, subtrahends):
        self.minuend = minuend
        self.subtrahends = subtrahends
    def __repr__(self):
        return 'Difference('+repr(self.minuend)+', '+', '.join([repr(subtrahend) for subtrahend in self.subtrahends])+')'
    def __str__(self):
        return '%s-%s' % (str(self.minuend), '-'.join(('(%s)' if isprecop(subtrahend, self.__class__) else '%s') % str(subtrahend) for subtrahend in self.subtrahends))
    def eval(self, vars, funcs):
        minuend = self.minuend.eval(vars, funcs)
        nums, subtrahends = splitnums([subtrahend.eval(vars, funcs) for subtrahend in self.subtrahends])
        if len(nums) > 0:
            num = math.fsum(nums)
            if isinstance(minuend, Number):
                minuend = Number(float(minuend) - num)
            elif num != 0:
                subtrahends.append(Number(num))
        if len(subtrahends) == 0:
            return minuend
        if isinstance(minuend, Number) and float(minuend) == 0:
            if len(subtrahends) == 1:
                return Negation(subtrahends[0])
            else:
                return Difference(Negation(subtrahends[0]), subtrahends[1:])
        if len(subtrahends) == 1:
            return subtrahends[0]
        return Difference(minuend, subtrahends)
    def contains(self, name):
        #return reduce(lambda x, y: x or y.contains(name), self.subtrahends, self.minuend.contains(name))
        for subtrahend in self.subtrahends:
            if subtrahend.contains(name):
                return True
        return False or self.minuend.contains(name)
    def unapply(self, funcs):
        self.minuend = self.minuend.unapply(funcs)
        self.subtrahends = [subtrahend.unapply(funcs) for subtrahend in self.subtrahends]
        return self
    def apply(self, name, term):
        self.minuend = self.minuend.apply(name, term)
        self.subtrahends = [subtrahend.apply(name, term) for subtrahend in self.subtrahends]
        return self

class Product(Term):
    def __init__(self, factors):
        self.factors = factors
    def __repr__(self):
        return 'Product('+', '.join([repr(factor) for factor in self.factors])+')'
    def __str__(self):
        return '*'.join([('(%s)' if isprecop(factor, self.__class__) else '%s') % str(factor) for factor in self.factors])
    def eval(self, vars, funcs):
        nums, factors = splitnums([factor.eval(vars, funcs) for factor in self.factors])
        factors = reduce(lambda agg, term: agg+term.factors if isinstance(term, Product) else agg+[term], factors, [])
        if len(nums) == 0:
            return Product(factors)
        for factor in nums:
            if float(factor) == 0:
                return Number(0)
        num = reduce(lambda x, y: x*float(y), nums, 1)
        if len(factors) == 0:
            return Number(num)
        if num != 1:
            factors.append(Number(num))
        if len(factors) == 1:
            return factors[0]
        return Product(factors)
    def contains(self, name):
        for factor in self.factors:
            if factor.contains(name):
                return True
        return False
    def unapply(self, funcs):
        self.factors = [factor.unapply(funcs) for factor in self.factors]
        return self
    def apply(self, name, term):
        self.factors = [factor.apply(name, term) for factor in self.factors]
        return self

class Quotient(Term):
    def __init__(self, dividend, divisors):
        self.dividend = dividend
        self.divisors = divisors
    def __repr__(self):
        return 'Quotient('+repr(self.dividend)+', '+', '.join([repr(divisor) for divisor in self.divisors])+')'
    def __str__(self):
        return '%s/%s' % (str(self.dividend), '/'.join([('(%s)' if isprecop(divisor, self.__class__) else '%s') % str(divisor) for divisor in self.divisors]))
    def eval(self, vars, funcs):
        dividend = self.dividend.eval(vars, funcs)
        if isinstance(dividend, Number) and float(dividend) == 0:
            return Number(0)
        nums, divisors = splitnums([divisor.eval(vars, funcs) for divisor in self.divisors])
        if len(nums) > 0:
            num = reduce(lambda x, y: x*float(y), nums, 1)
            if isinstance(dividend, Number):
                dividend = Number(float(dividend)/num)
            else:
                divisors.append(Number(num))
        if len(divisors) == 0:
            return dividend
        return Quotient(dividend, divisors)
    def contains(self, name):
        for divisor in self.divisors:
            if divisor.contains(name):
                return True
        return False or self.dividend.contains(name)
    def unapply(self, funcs):
        self.dividend = self.dividend.unapply(funcs)
        self.divisors = [divisor.unapply(funcs) for divisor in self.divisors]
        return self
    def apply(self, name, term):
        self.dividend = self.dividend.apply(name, term)
        self.divisors = [divisor.apply(name, term) for divisor in self.divisors]
        return self

class Power(Term):
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp
    def __repr__(self):
        return 'Power('+repr(self.base)+', '+repr(self.exp)+')'
    def __str__(self):
        base = ('(%s)' if isprecop(self.base, self.__class__) else '%s') % str(self.base)
        exp = ('(%s)' if isprecop(self.exp, self.__class__) else '%s') % str(self.exp)
        return '%s**%s' % (base, exp)
    def eval(self, vars, funcs):
        base = self.base.eval(vars, funcs)
        exp = self.exp.eval(vars, funcs)
        if isinstance(base, Number) and isinstance(exp, Number):
            return Number(float(base)**float(exp))
        elif isinstance(exp, Number) and float(exp) == 0:
            return Number(1)
        elif isinstance(base, Number) and float(base) == 0:
            return Number(0)
        return Power(base, exp)
    def contains(self, name):
        return self.base.contains(name) or self.exp.contains(name)
    def unapply(self, funcs):
        self.base = self.base.unapply(funcs)
        self.exp = self.exp.unapply(funcs)
        return self
    def apply(self, name, term):
        self.base = self.base.apply(name, term)
        self.exp = self.exp.apply(name, term)
        return self

class Negation(Term):
    def __init__(self, term):
        self.term = term
    def __repr__(self):
        return 'Negation(%s)' % repr(self.term)
    def __str__(self):
        return '-%s' % (('(%s)' if isprecop(self.term, self.__class__) else '%s') % str(self.term))
    def eval(self, vars, funcs):
        term = self.term.eval(vars, funcs)
        if isinstance(term, Number):
            return Number(-float(term))
        if isinstance(term, Negation):
            return term.term
        return Negation(term)
    def contains(self, name):
        return self.term.contains(name)
    def unapply(self, funcs):
        self.term = self.term.unapply(funcs)
        return self
    def apply(self, name, term):
        self.term = self.term.apply(name, term)
        return self

class VariableAssignment(Term):
    def __init__(self, name, term):
        self.name = name
        self.term = term
    def __repr__(self):
        return 'VariableAssignment('+repr(self.name)+', '+repr(self.term)+')'
    def __str__(self):
        return str(self.name)+' = '+str(self.term)
    def eval(self, vars, funcs):
        term = self.term.eval(vars, funcs)
        if isinstance(term, Number):
            vars[self.name] = term
        return VariableAssignment(self.name, term)
    def contains(self, name):
        return self.term.contains(name)
    def unapply(self, funcs):
        self.term = self.term.unapply(funcs)
        return self
    def apply(self, name, term):
        self.term = self.term.apply(name, term)
        return self

class FunctionAssignment(Term):
    def __init__(self, name, args, term):
        self.name = name
        self.args = tuple(args)
        self.term = term
    def __repr__(self):
        return 'FunctionAssignment('+repr(self.name)+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), '+repr(self.term)+')'
    def __str__(self):
        return '%s(%s) = %s' % (str(self.name), ', '.join([str(arg) for arg in self.args]), str(self.term))
    def eval(self, vars, funcs):
        term = self.term.eval(vars, funcs)
        if isinstance(term, Number):
            funcs[self.name] = (self.args, term)
        return FunctionAssignment(self.name, self.args, term)
    def contains(self, name):
        return self.term.contains(name)
    def unapply(self, funcs):
        self.term = self.term.unapply(funcs)
        return self
    def apply(self, name, term):
        self.term = self.term.apply(name, term)
        return self

class Variable(Term):
    def __init__(self, name):
        self.name = str(name)
    def __repr__(self):
        return 'Variable('+repr(self.name)+')'
    def __str__(self):
        return self.name
    def eval(self, vars, funcs):
        if self.name in vars and vars[self.name] != None:
            return vars[self.name]
        else:
            return self
    def contains(self, name):
        return self.name == name
    def unapply(self, funcs):
        if self.name in funcs:
            args = funcs[self.name]
            arglist = [Variable(arg) for arg in args]
            if len(arglist) > 0:
                return Function(self.name, arglist)
        return self
    def apply(self, names, terms):
        for name, term in zip(names, terms):
            if self.name == name:
                return term
        return self

class Function(Term):
    def __init__(self, name, args):
        self.name = str(name)
        self.args = tuple(args)
    def __repr__(self):
        return 'Function('+self.name+', Arguments('+', '.join([repr(arg) for arg in self.args])+'))'
    def __str__(self):
        return '%s(%s)' % (self.name, ', '.join([str(arg) for arg in self.args]))
    def eval(self, vars, funcs):
        if self.name in funcs and funcs[self.name] != None:
            args, term = funcs[self.name]
            return reduce(lambda x, y: x.apply(y, vars[y]) if y in vars else x, args, term)
        else:
            return Function(self.name, [arg.eval(vars, funcs) for arg in self.args])
    def contains(self, name):
        for arg in self.args:
            if arg.contains(name):
                return True
        return False or self.name == name
    def unapply(self, funcs):
        if self.name in funcs:
            if len(self.args) < len(funcs[self.name]):
                args = self.args+funcs[self.name][len(self.args):]
                return Function(self.name, args)
        return self
    def apply(self, name, term):
        if self.name == name:
            return term
        self.args = [arg.apply(name, term) for arg in self.args]
        return self

opprec[Sum] = ()
opprec[Difference] = (Sum, Difference)
opprec[Product] = (Sum, Difference)
opprec[Quotient] = (Sum, Difference, Product, Quotient)
opprec[Power] = (Sum, Difference, Product, Quotient)
opprec[Negation] = (Sum, Difference, Product, Quotient, Power)

class Plotter():
    def __init__(self, filename=None, inputs={}, integrate=False, intargs=('x', 'y'), intstart=0.0, intend=1.0, intstops=5):
        intargs = intargs # Variables over which to integrate numerically.
        filters = []
        #filters = ['Function 5']
        if filename:
            try:
                with open(filename, 'r') as f:
                    fstr = f.read()
                    inputs = re.findall(r'"\[(?:.*?)\]"', fstr, re.S)
            except:
                raise
        if inputs and type(inputs) == list:
            inputs = {'Function %d' % (i+1): input for i, input in enumerate(inputs)}
        self.parsers = {name: Parser(name, input, integrate, intargs, intstart, intend, intstops) for name, input in inputs.items() if not filters or name in filters} # One Parser object for each input string to be parsed.

    def setvar(self, *args):
        self.setvars(args)

    def setvars(self, *args):
        [self.parsers[name].setvars(*args) for name in self.parsers]

    def getvars(self, sort=False):
        vars = list(set(flatten([self.parsers[name].getvars() for name in self.parsers])))
        if sort:
            vars.sort()
        return vars

    def settings(self, **kwargs):
        [self.parsers[name].settings(**kwargs) for name in self.parsers]

    def gp(self, command, name=None):
        [self.parsers[name].gp(command) for name in self.parsers]

    def plot(self, *args):
        [self.parsers[name].plot(*args) for name in self.parsers]

    def replot(self):
        [self.parsers[name].replot() for name in self.parsers]

    def export(self, foldername):
        [self.parsers[name].export(foldername) for name in self.parsers]

class Parser():
    def __init__(self, name=None, input=None, integrate=False, intargs=('x', 'y'), intstart=0.0, intend=1.0, intstops=5):
        self.name = name
        self.integrate = integrate # Sets default integration mode.
        self.intargs = makeargs(intargs, sort=True) # Sets default variables over which to integrate numerically.
        self.intstart = intstart # Sets default integration start.
        self.intend = intend # Sets default integration end.
        self.intstops = intstops # Sets default number of horizontal intergration parts.

        self.exprlist = [] # Contains the parsed result of the main input string.
        self.vardict = {} # Variable definitions, used to look up values for string generation.
        self.funclists = {} # Simplified exprlist, with unapplied functions over certain arguments.

        # Set plot-evaluating variables
        self.intvals = None
        self.sargs = None
        self.plotvars = None
        self.plotexprs = []

        self.gp = Gnuplot(debug=0)
        if input:
            self.parse(input)

    def log(self, msg):
        print('Parser %s: %s' % (self.name, msg))

    def settings(self, name=None, args=None, integrate=None, intargs=None, intstart=None, intend=None, intstops=None):
        """Function to set object-wide settings for certain variables."""
        self.name = name if name != None else self.name
        self.args = makeargs(args, sort=True) if args != None else self.args
        self.intargs = makeargs(intargs, sort=True) if intargs != None else self.intargs
        self.integrate = integrate if integrate != None else self.integrate
        self.intargs = intargs if intargs != None else self.intargs
        self.intstart = intstart if intstart != None else self.intstart
        self.intend = intend if intend != None else self.intend
        self.intstops = intstops if intstops != None else self.intstops

    def tokenize(self, input):
        """Takes a string input of an expression of the supported types and returns a tokenized list of single terms. Results should be passed to realize() to obtain a parsed expression tree."""
        current = ''
        mode = ''
        newmode = ''
        result = [[]]
        for c in input:
            if (c in nums and mode != 'var') or (c in ['e', 'E'] and mode == 'num'):
                newmode = 'num'
            elif c in alpha or (c in nums and mode == 'var'):
                if mode == 'num':
                    mode == 'break'
                newmode = 'var'
            elif c in ops:
                if c == '-' and (isop(current) or (current == '' and (result[-1] == [] or isop(result[-1][-1])))):
                    newmode = 'adapt'
                else:
                    newmode = 'ops'
            elif c in seps:
                newmode = 'seps'
            elif c == ' ':
                newmode = 'break'
            elif c == '(':
                if current.strip() != '':
                    result[-1].append(current)
                current = ''
                newmode = ''
                result.append([])
                continue
            elif c == ')':
                if current.strip() != '':
                    result[-1].append(current)
                result[-2].append(result.pop())
                current = ''
                newmode = ''
                continue
            else:
                newmode = ''
                self.log('Unhandled character: %s' % c)
            if not newmode == 'adapt' and (newmode != mode or newmode == 'break'):
                if current.strip() != '':
                    result[-1].append(current)
                current = c
                mode = newmode
            else:
                current += c
        if current.strip() != '':
            result[-1].append(current)
        return result.pop()

    def classify(self, tlist):
        """Classifies list elements into categories (Number, Variable, etc.)."""
        parsed = []
        opdict = {op: [] for op in oplist}

        # Classification of single tokens
        offset = 0
        for i in range(len(tlist)):
            tok = tlist[i]
            if isnum(tok):
                parsed.append(Number(tok))
            elif isvar(tok):
                parsed.append(Variable(tok))
                if tok not in self.vardict:
                    self.vardict[tok] = None
            elif islist(tok):
                if i > 0 and isinstance(parsed[-1], Variable):
                    parsed[-1] = Function(parsed[-1].name, self.classifyarguments(tok))
                    del self.vardict[parsed[-1].name]
                    offset += 1
                else:
                    parsed.append(self.classify(tok))
            elif isop(tok):
                opdict[tok].append(i-offset)
                parsed.append(tok)
            else:
                parsed.append(tok)
            if len(parsed) > 1 and ((isinstance(parsed[-2], Term) and isinstance(parsed[-1], Term))):
                parsed[-1:] = ['*', parsed[-1]]
                opdict['*'].append(i-offset)
                offset -= 1

        # Classification of operators and combined tokens
        offset = 0
        for i in opdict['-']:
            try:
                i -= offset
                if i < 0 or isop(parsed[i+1]):
                    raise InvalidFormatError(deepjoin(tlist), "Negaion is a unary operator, requires one argument to the right.")
                if i == 0 or isop(parsed[i-1]):
                    parsed[i:i+2] = [Negation(parsed[i+1])]
                    del opdict['-'][i+offset]
                    offset += 1
                    for key in opdict:
                        opdict[key] = [n-1 if n > i else n for n in opdict[key]]
            except IndexError:
                raise InvalidFormatError(deepjoin(tlist), "Negation is a unary operator, requires one argument to the right.")
        offset = 0
        for i in sorted(opdict['**']+opdict['^']):
            try:
                i -= offset
                if i <= 0 or isop(parsed[i-1]) or isop(parsed[i+1]):
                    raise InvalidFormatError(deepjoin(tlist), "Exponentiation is a binary operator, requires one argument to the left and one to the right.")
                parsed[i-1:i+2] = [Power(parsed[i-1], parsed[i+1])]
                offset += 2
                for key in opdict:
                    opdict[key] = [n-2 if n > i else n for n in opdict[key]]
            except IndexError:
                raise InvalidFormatError(deepjoin(tlist), "Exponentiation is a binary operator, requires one argument to the left and one to the right.")
        offset = 0
        for i in sorted(opdict['*']+opdict['/']):
            try:
                i -= offset
                if i <= 0 or isop(parsed[i-1]) or isop(parsed[i+1]):
                    if i in opdict['*']:
                        raise InvalidFormatError(deepjoin(tlist), "Multiplication is a binary operator and requires one argument to the left and one to the right.")
                    elif i in opdict['/']:
                        raise InvalidFormatError(deepjoin(tlist), "Division is a binary operator and requires one argument to the left and one to the right.")
                if i in opdict['*']:
                    if isinstance(parsed[i-1], Product):
                        parsed[i-1].factors.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Product([parsed[i-1], parsed[i+1]])]
                elif i in opdict['/']:
                    if isinstance(parsed[i-1], Quotient):
                        parsed[i-1].divisors.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Quotient(parsed[i-1], [parsed[i+1]])]
                offset += 2
                for key in opdict:
                    opdict[key] = [n-2 if n >= i else n for n in opdict[key]]
            except IndexError:
                if i in opdict['*']:
                    raise InvalidFormatError(deepjoin(tlist), "Multiplication is a binary operator and requires one argument to the left and one to the right.")
                elif i in opdict['/']:
                    raise InvalidFormatError(deepjoin(tlist), "Division is a binary operator and requires one argument to the left and one to the right.")
        offset = 0
        for i in sorted(opdict['+']+opdict['-']):
            try:
                i -= offset
                if i <= 0 and isop(parsed[i-1]) or i < 0 and isop(parsed[i+1]):
                    if i in opdict['+']:
                        raise InvalidFormatError(deepjoin(tlist), "Addition is a binary operator and requires one argument to the left and one to the right.")
                    elif i in opdict['-']:
                        raise InvalidFormatError(deepjoin(tlist), "Subtraction is a binary operator and requires one argument to the left and one to the right.")
                if i in opdict['+']:
                    if isinstance(parsed[i-1], Sum):
                        parsed[i-1].summands.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Sum([parsed[i-1], parsed[i+1]])]
                elif i in opdict['-']:
                    if isinstance(parsed[i-1], Difference):
                        parsed[i-1].subtrahends.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Difference(parsed[i-1], [parsed[i+1]])]
                offset += 2
                for key in opdict:
                    opdict[key] = [n-2 if n >= i else n for n in opdict[key]]
            except IndexError:
                if i in opdict['+']:
                    raise InvalidFormatError(deepjoin(tlist), "Addition is a binary operator and requires one argument to the left and one to the right.")
                elif i in opdict['-']:
                    raise InvalidFormatError(deepjoin(tlist), "Subtraction is a binary operator and requires one argument to the left and one to the right.")
        try:
            return parsed[0]
        except IndexError:
            if len(tlist) == 0:
                raise InvalidFormatError(deepjoin(tlist), "Empty input list.")
            raise InvalidFormatError(deepjoin(tlist), "Unknown format error.")

    def setintargs(self, *args):
        """Sets intargs to whatever is passed as arguments, formatted by makeargs()."""
        self.intargs = makeargs(args, sort=True)

    def classifyarguments(self, l):
        """Sub-function to create a list of arguments, to not be confused with a parenthesized expression."""
        return [self.classify(arg) for arg in split(l, ',')]

    def realize(self, tlist):
        """Takes a tokenized list of terms and constructs an expression tree from it."""
        if '=' in tlist:
            if isvar(tlist[0]) and isass(tlist[1]) and len(tlist) > 2 and '=' not in tlist[2:]:
                return VariableAssignment(tlist[0], self.classify(tlist[2:]))
            elif isvar(tlist[0]) and islist(tlist[1]) and isass(tlist[2]) and len(tlist) > 3 and '=' not in tlist[3:]:
                return FunctionAssignment(tlist[0], self.classifyarguments(tlist[1]), self.classify(tlist[3:]))
            else:
                if tlist.count('=') > 1:
                    reason = "Nested or multiple assignments are not allowed."
                elif not isvar(tlist[0]):
                    reason = "Only variables and functions can be assigned to."
                elif tlist[-1] == '=':
                    reason = "Empty assignments are not allowed."
                else:
                    reason = "Unkown format error."
                raise InvalidFormatError(deepjoin(tlist), reason)
        return self.classify(tlist)

    def unapply(self, *args):
        """Turns all variables into functions over args, if any args appear within the variable."""
        exprs = deepcopy(self.exprlist)
        funcdict = {}
        for i in range(len(exprs)):
            expr = exprs[i]
            if isinstance(expr, (VariableAssignment, FunctionAssignment)):
                expr.unapply(funcdict)
                arglist = makeargs(filter(lambda el: expr.contains(el), args))
                arglist = tuple([self.classify([arg]) for arg in arglist])
                if len(arglist) > 0:
                    if isinstance(expr, FunctionAssignment):
                        arglist = makeargs(expr.args+arglist)
                    expr = FunctionAssignment(expr.name, arglist, expr.term)
                    exprs[i] = expr
                else:
                    expr = VariableAssignment(expr.name, expr.term)
                funcdict[expr.name] = arglist
        return exprs

    def unapplyall(self, vars):
        """Unapplies all parsed terms w.r.t. all combinations of arguments provided in *args. Results are stored for further application."""
        #vars = [var for var in vars if not self.integrate or var not in self.intargs]
        combinations = set([makeargs((str(var1), str(var2)), sort=True) for var1 in vars for var2 in vars])
        for arglist in combinations:
            self.funclists[filter(lambda arg: arg not in self.intargs, arglist)] = self.unapply(*arglist+self.intargs)

    def numintegrate(self, expr, integrate, intargs, start, end, stops):
        """Numerical integration of expr, w.r.t. args and at the points provided in subs."""
        length = math.fabs(end-start)/stops
        subs = None
        area = None
        if type(integrate) == bool and integrate:
            if len(intargs) == 2:
                integrate = 'RTriangle'
            elif len(intargs) == 1:
                integrate = 'Line'
            else:
                integrate = 'Self'
                self.log('None of the integral variables found.')
        if integrate == 'RTriangle':
            subs = filter(lambda (x, y): x+y <= end, map(lambda (x, y): (start+x*length, start+y*length), flatten([map(lambda (x, y): (x+offset, y+offset), [(i, j) for i in range(stops) for j in range(stops-i)]) for offset in [1.0/3.0, 2.0/3.0]], level=1)))
            area = length**2/2.0
        elif integrate == 'Line':
            subs = [(x+0.5)/length for x in range(stops)]
            area = length
        elif integrate == 'Self':
            subs = []
            area = 1
        if not integrate == 'Self' and (subs == None or area == None):
            if type(integrate) == str:
                raise InvalidOperationError('Integration domain %s unknown or not supported.' % self.integrate)
            else:
                raise InvalidOperationError('No integration domain specified.')
        subs = [map(self.classify, [[term] for term in sub]) for sub in subs]
        def expand(term):
            if integrate == 'Self':
                return term
            return Product([Number(area), Sum([Function('abs', [deepcopy(term).apply(intargs, sub)]) for sub in subs])])
        if isinstance(expr, (VariableAssignment, FunctionAssignment)):
            if isinstance(expr, FunctionAssignment):
                expr.args = filter(lambda arg: str(arg) not in intargs, expr.args)
                if len(expr.args) == 0:
                    expr = VariableAssignment(expr.name, expr.term)
            expr.term = expand(expr.term)
        else:
            expr = expand(expr)
        return expr

    def formatinput(self, input):
        """Formats Maple output to a list of separate expressions"""
        lines = input.strip().splitlines()
        terms = []
        term = ''
        total = len(lines)
        nestlevel = 0
        for i in range(total):
            line = lines[i].strip()
            if line == '' or line.startswith('>'):
                continue
            if line.startswith('"['):
                line = line[2:]
            if line.endswith(']"'):
                line = line[:-2]
            elif line.endswith('\\'):
                line = line[:-1]
            elif i < total-1:
                line = line+" "
            line = line.replace('[', '_').replace(']', '')
            for c in line:
                if c == '(':
                    nestlevel += 1
                elif c == ')':
                    nestlevel -= 1
                    if nestlevel < 0:
                        raise InvalidFormatError(input, "Parenthesis mismatch.")
                elif c == ' ' and term == '':
                    continue
                if c == ',' and nestlevel == 0:
                    terms.append(term)
                    term = ''
                else:
                    term += c
        terms.append(term)
        if nestlevel > 0:
            raise InvalidFormatError(input, "Parenthesis mismatch.")
        return terms

    def parseterm(self, term):
        """Parses a single term into an internal expression"""
        #self.log(term)
        #self.log(deepjoin(self.tokenize(term)))
        #self.log(str(self.realize(self.tokenize(term))).replace('.0', '').replace('**', '^'))
        #self.log('Checks:')
        #self.log(term == deepjoin(self.tokenize(term)))
        #self.log(term == str(self.realize(self.tokenize(term))).replace('.0', '').replace('**', '^'))
        return self.realize(self.tokenize(term))

    def parselist(self, l):
        """Takes a list of terms and parses them."""
        return [self.parseterm(el) for el in l]

    def parse(self, input, unapply=True):
        """Parse a comma-separated list of terms."""
        self.exprlist = self.parselist(self.formatinput(input))
        for expr in self.exprlist:
            if isinstance(expr, (VariableAssignment, FunctionAssignment)):
                try:
                    del self.vardict[expr.name]
                except KeyError:
                    pass
        if unapply:
            self.unapplyall(self.vardict)
        #self.exprlist = [expr for expr in self.exprlist if self.exprlist[-1].contains(expr.name) or expr == self.exprlist[-1]]

    def setvar(self, *args):
        """Packs the variables from the argument tuple and sets it to setvars as a list of length one."""
        self.setvars(args)

    def setvars(self, *args):
        """Sets a variable to the specified value. When that variable appears in an expression, it will resolve to the value instead of its name."""
        args = {name: self.parseterm(str(val)) for name, val in args}
        self.vardict.update(args)

    def getvars(self, sort=False):
        """Returns a list of currently tracked variables."""
        varlist = [var for var in self.vardict if not self.integrate or self.integrate and var not in self.intargs]
        if sort:
            varlist.sort()
        return tuple(varlist)

    def simplify(self, exprs):
        """Takes a list of expressions and simplifies them much as possible with considerations to set variables. Returns a (possibly shortened) list of expressions."""
        exprs = deepcopy(exprs)
        valdict = {}
        offset = 0
        self.log('Before: %s' % str(exprs))
        for i in range(len(exprs)):
            i -= offset
            try:
                expr = exprs[i]
                try:
                    for name, value in valdict:
                        expr.apply(name, value)
                    valdict[expr.name] = float(str(expr.term))
                    exprs[i:i+1] = []
                    offset += 1
                except ValueError:
                    continue
            except IndexError:
                break
        self.log('After: %s' % str(exprs))

    def evalfuncs(self, sargs):
        vardict = {name: val for name, val in deepcopy(self.vardict).items() if name not in sargs and (not self.integrate or name not in self.intargs)}
        funcdict = {}
        exprs = deepcopy(self.funclists[sargs])
        return [expr.eval(vardict, funcdict) for expr in deepcopy(self.funclists[sargs])]

    def plot(self, args=None, integrate=None, intargs=None, intstart=None, intend=None, intstops=None):
        """Plots the parsed expression list with specified settings."""
        args = makeargs(args) if args != None else self.args
        self.args = args
        if len(args) < 1 or len(args) > 2:
            raise InvalidCallError('(%s)' % ', '.join(args), 'Invalid number of arguments. One argument for a 2D plot, two for a 3D plot.')

        # Settings passed manually should override previously stored or default settings.
        intargs = makeargs(intargs) if intargs != None else self.intargs
        integrate = integrate if integrate != None else self.integrate
        intargs = intargs if intargs != None else self.intargs
        intstart = intstart if intstart != None else self.intstart
        intend = intend if intend != None else self.intend
        intstops = intstops if intstops != None else self.intstops

        sargs = makeargs(args, sort=True)
        reapply = sargs != self.sargs
        reeval = reapply or self.plotvars != {name: str(val) for name, val in self.vardict.items()}
        reintegrate = integrate and (reapply or reeval or (integrate, intargs, intstart, intend, intstops) != self.intvals)

        # Unapply expression list for specified arguments, if not already done.
        if reapply:
            if sargs not in self.funclists:
                self.funclists[sargs] = self.unapply(*args+intargs)
            self.sargs = sargs

        # Format the parsed expression list if it hasn't been previously formatted with the current settings
        if reeval:
            self.plotexprs = self.evalfuncs(sargs)
            self.plotvars = {name: str(val) for name, val in self.vardict.items()}

        # Integrate, if specified and not already integrated with current variables.
        if integrate:
            plotexpr = self.numintegrate(deepcopy(self.plotexprs[-1]), integrate, intargs, intstart, intend, intstops)
            self.intvals = (integrate, intargs, intstart, intend, intstops)
        else:
            plotexpr = self.plotexprs[-1]

        commands = []

        # Set plotting variables.
        if len(args) == 1:
            commands.append('unset pm3d')
            commands.append('unset ylabel')
            commands.append('set xlabel "%s"' % args[0])
            commands.append('set autoscale')
            commands.append('set dummy %s' % args[0])
        elif len(args) == 2:
            commands.append('set xlabel "%s"' % args[0])
            commands.append('set ylabel "%s"' % args[1])
            commands.append('set pm3d')
            commands.append('set dummy %s, %s' % args)

        commands.extend([str(expr) for expr in self.plotexprs if isinstance(expr, (VariableAssignment, FunctionAssignment))])
        if isinstance(plotexpr, (VariableAssignment, FunctionAssignment)):
            commands[-1] = str(plotexpr)

        # Determine the plot command based on variables and the last expression.
        plotcmd = '%s %%s ls 1 title "%s"' % ('splot' if len(args) == 2 else 'plot', self.name)
        if isinstance(plotexpr, FunctionAssignment):
            plotargs = [str(parg) for parg in plotexpr.args]
            fargs = filter(lambda arg: arg in plotargs, args)
            commands.append(plotcmd % ('%s(%s)' % (plotexpr.name, ', '.join(fargs))))
        elif isinstance(plotexpr, VariableAssignment):
            commands.append(plotcmd % plotexpr.name)
        else:
            commands.append(plotcmd % plotexpr)

        #self.log('Gnuplot commands:\n%s\n' % '\n'.join([command for command in commands]))

        # Send the formatted commands to Gnuplot instance.
        [self.gp(command) for command in commands]

    def replot(self):
        if self.sargs:
            self.gp('replot')

    def export(self, foldername):
        filename = self.name
        self.gp('set term pdfcairo size 5.0in,3.0in')
        self.gp('set output "%s/%s.pdf"' % (foldername, filename))
        self.replot()
        self.gp('set term wxt')

if __name__ == '__main__':
    #termstring = 't1 = 0.5*3*h[1]+u[1]*h[2], t2 = 2*t1 + 3/(h[1]*h[2]*u[2]), t3 = 2h[1]+4t2 u[1], t4 = 2 h[1]**2 + t2^3 + t3*u[2]+ 5 t1 *(h[1]*h[2]), t15= t4*u[1]*u[2]*h[1]^2*h[2]**2 +  h[1], t1642 = 0.12t15+h[1]'
    termstring1 = 't1 = 0.5*3*h[1]+x*u[1]*h[2], t2 = y+2*t1 + 3/(h[1]*h[2]*u[2]), t3(h[2]) = t2 + h[2]*x + u[2]*y'
    termstring2 = 't1 = 2, t3 = 2*h[1]+x, t65 = u[1]**2, t122 = y + t65*t3^t1'
    print('Not supposed to be run on its own. Demonstrative run using the following inputs:\nA-Term:\n%s\n\nTerm Ber:\n%s\n' % ('\n'.join(termstring1.split(', ')), '\n'.join(termstring2.split(', '))))

    #p = Plotter({'A-Term': termstring1, 'Term Ber': termstring2}, integrate=True)
    #p1 = p.parsers['A-Term']
    #p2 = p.parsers['Term Ber']

    #p.setvars(('h_1', 5), ('h_2', 3.1), ('u_1', 4), ('u_2', -0.2))
    #p.plot(('h_1', 'h_2'))

    p = Parser()

    import readline
    import code

    vars = globals().copy()
    vars.update(locals())

    shell = code.InteractiveConsole(vars)
    shell.interact()
