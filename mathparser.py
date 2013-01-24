from copy import deepcopy
import math
import sys

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

class Term():
    def contains(self, name):
        return False
    def setvar(self, name, val):
        pass
    def unapply(self, name, *args):
        return self
    def apply(self, name, term):
        return self

class Number(Term):
    def __init__(self, number):
        try:
            self.number = float(number)
        except:
            print("String %s mistakenly parsed as number." % number)
            raise
    def __repr__(self):
        return 'Number('+repr(self.number)+')'
    def __str__(self):
        return str(self.number)

class Sum(Term):
    def __init__(self, summands):
        self.summands = summands
    def __repr__(self):
        return 'Sum('+', '.join([repr(summand) for summand in self.summands])+')'
    def __str__(self):
        summands = []
        nums = []
        for summand in [str(summand) for summand in self.summands]:
            try:
                nums.append(float(summand))
            except ValueError:
                summands.append(summand)
        if len(nums) == 0:
            return '+'.join(summands)
        if len(summands) == 0:
            return str(math.fsum(nums))
        return '%s+%s' % (str(math.fsum(nums)), '+'.join(summands))
    def contains(self, name):
        for summand in self.summands:
            if summand.contains(name):
                return True
        return False
    def setvar(self, name, val):
        for summand in self.summands:
            summand.setvar(name, val)
    def unapply(self, name, *args):
        self.summands = [summand.unapply(name, *args) for summand in self.summands]
        return self
    def apply(self, name, term):
        self.summands = [summand.apply(name, term) for summand in self.summands]
        return self

class Difference(Term):
    def __init__(self, subtrahend, minuends):
        self.subtrahend = subtrahend
        self.minuends = minuends
    def __repr__(self):
        return 'Difference('+repr(self.subtrahend)+', '+', '.join([repr(minuend) for minuend in self.minuends])+')'
    def __str__(self):
        minuends = []
        nums = []
        for minuend in [str(minuend) for minuend in self.minuends]:
            try:
                nums.append(float(minuend))
            except ValueError:
                minuends.append(minuend)
        if len(nums) == 0:
            return '%s-%s' % (str(self.subtrahend), '-'.join(minuends))
        subtrahend = str(self.subtrahend)
        try:
            subtrahend = str(float(subtrahend) - math.fsum(nums))
        except ValueError:
            minuends.append(str(math.fsum(nums)))
        if len(minuends) == 0:
            return subtrahend
        return '%s-%s' % (subtrahend, '-'.join(minuends))
    def contains(self, name):
        for minuend in self.minuends:
            if minuend.contains(name):
                return True
        return False or self.subtrahend.contains(name)
    def setvar(self, name, val):
        self.subtrahend.setvar(name, val)
        for minuend in self.minuends:
            minuend.setvar(name, val)
    def unapply(self, name, *args):
        self.subtrahend = self.subtrahend.unapply(name, *args)
        self.minuends = [minuend.unapply(name, *args) for minuend in self.minuend]
        return self
    def apply(self, name, term):
        self.subtrahend = self.subtrahend.apply(name, term)
        self.minuends = [minuend.apply(name, term) for minuend in self.minuends]
        return self

class Product(Term):
    def __init__(self, factors):
        self.factors = factors
    def __repr__(self):
        return 'Product('+', '.join([repr(factor) for factor in self.factors])+')'
    def __str__(self):
        factors = []
        nums = []
        for factor in [('(%s)' if isinstance(factor, (Sum, Difference)) else '%s') % str(factor) for factor in self.factors]:
            try:
                nums.append(float(factor))
            except ValueError:
                factors.append(factor)
        if len(nums) == 0:
            return '*'.join(factors)
        if len(factors) == 0:
            return str(math.fsum(nums))
        return '%s*%s' % (str(math.fsum(nums)), '*'.join(factors))
    def contains(self, name):
        for factor in self.factors:
            if factor.contains(name):
                return True
        return False
    def setvar(self, name, val):
        for factor in self.factors:
            factor.setvar(name, val)
    def unapply(self, name, *args):
        self.factors = [factor.unapply(name, *args) for factor in self.factors]
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
        dividend = ('(%s)' if isinstance(self.dividend, (Sum, Difference)) else '%s') % str(self.dividend)
        divisors = []
        nums = []
        for divisor in [('(%s)' if isinstance(divisor, (Sum, Difference, Product)) else '%s') % str(divisor) for divisor in self.divisors]:
            try:
                nums.append(float(divisor))
            except ValueError:
                divisors.append(divisor)
        if len(nums) == 0:
            return '%s/%s' % (str(self.dividend), '/'.join(divisors))
        dividend = str(self.dividend)
        try:
            dividend = str(float(dividend) / reduce(lambda x, y: x*y, nums))
        except ValueError:
            divisors.append(str(reduce(lambda x, y: x*y, nums)))
        if len(divisors) == 0:
            return dividend
        return '%s/%s' % (dividend, '/'.join(divisors))
    def contains(self, name):
        for divisor in self.divisors:
            if divisor.contains(name):
                return True
        return False or self.dividend.contains(name)
    def setvar(self, name, val):
        self.dividend.setvar(name, val)
        for divisor in self.divisors:
            divisor.setvar(name, val)
    def unapply(self, name, *args):
        self.dividend = self.dividend.unapply(name, *args)
        self.divisors = [divisor.unapply(name, *args) for divisor in self.divisors]
        return self
    def apply(self, name, term):
        self.dividend = self.dividend.apply(name, term)
        self.divisors = [divisor.apply(name, term) for divisor in self.divisors]
        return self

class Exponent(Term):
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp
    def __repr__(self):
        return 'Exponent('+repr(self.base)+', '+repr(self.exp)+')'
    def __str__(self):
        base = ('(%s)' if isinstance(self.base, (Sum, Difference, Product, Quotient)) else '%s') % str(self.base)
        exp = ('(%s)' if isinstance(self.exp, (Sum, Difference, Product, Quotient)) else '%s') % str(self.exp)
        try:
            return str(float(base)**float(exp))
        except ValueError:
            return '%s**%s' % (base, exp)
    def contains(self, name):
        return self.base.contains(name) or self.exp.contains(name)
    def setvar(self, name, val):
        self.base.setvar(name, val)
        self.exp.setvar(name, val)
    def unapply(self, name, *args):
        self.base = self.base.unapply(name, *args)
        self.exp = self.exp.unapply(name, *args)
        return self
    def apply(self, name, term):
        self.base = self.base.apply(name, term)
        self.exp = self.exp.apply(name, term)
        return self

class VariableAssignment(Term):
    def __init__(self, name, term):
        self.name = name
        self.term = term
    def __repr__(self):
        return 'VariableAssignment('+repr(self.name)+', '+repr(self.term)+')'
    def __str__(self):
        return str(self.name)+' = '+str(self.term)
    def contains(self, name):
        return self.term.contains(name)
    def setvar(self, name, val):
        self.term.setvar(name, val)
    def unapply(self, name, *args):
        self.term = self.term.unapply(name, *args)
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
    def contains(self, name):
        return self.term.contains(name)
    def setvar(self, name, val):
        self.term.setvar(name, val)
    def unapply(self, name, *args):
        self.term = self.term.unapply(name, *args)
        return self
    def apply(self, name, term):
        self.term = self.term.apply(name, term)
        return self

class Variable(Term):
    def __init__(self, name, val=None):
        self.name = str(name)
        self.val = val
    def __repr__(self):
        if self.val != None:
            return 'Variable('+repr(self.name)+', '+repr(self.val)+')'
        else:
            return 'Variable('+repr(self.name)+', Unassigned)'
    def __str__(self):
        if self.val != None:
            return str(self.val)
        else:
            return self.name
    def contains(self, name):
        return self.name == name
    def setvar(self, name, val):
        if self.name == name:
            self.val = val
    def unapply(self, name, *args):
        if self.name == name:
            arglist = [Variable(arg) for arg in args]
            return Function(self.name, arglist, self.val)
        return self
    def apply(self, names, terms):
        for name, term in zip(names, terms):
            if self.name == name:
                return term
        return self

class Function(Term):
    def __init__(self, name, args, val=None):
        self.name = str(name)
        self.args = args
        self.val = val
    def __repr__(self):
        if self.val != None:
            return 'Function('+self.name+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), '+repr(self.val)+')'
        else:
            return 'Function('+self.name+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), Unassigned)'
    def __str__(self):
        if self.val != None:
            return str(self.val)
        else:
            return '%s(%s)' % (self.name, ', '.join([str(arg) for arg in self.args]))
    def contains(self, name):
        for arg in self.args:
            if arg.contains(name):
                return True
        return False or self.name == name
    def setvar(self, name, val):
        if self.name == name:
            self.val = val
        for arg  in self.args:
            arg.setvar(name, val)
    def apply(self, name, term):
        if self.name == name:
            return term
        self.args = [arg.apply(name, term) for arg in self.args]
        return self

class Mathparser():
    def __init__(self):
        self.num = '0123456789.'
        self.alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        self.ops = '+-*/^'
        self.seps = '=,'
        self.oplist = ['+', '-', '*', '/', '^', '**']

        self.exprlist = []
        self.funclists = {} # Simplified exprlist, with unapplied functions over certain arguments.

    def split(self, l, sep):
        res = [[]]
        for el in l:
            if el == sep:
                res.append([])
            else:
                res[-1].append(el)
        return res

    def deepjoin(self, l):
        res = ''
        for el in l:
            if type(el) == list:
                res += '('+(self.deepjoin(el))+')'
            else:
                res += str(el)
        return res

    def isvar(self, tok):
        if type(tok) != str:
            return False
        if not tok[0] in self.alpha:
            return False
        for c in tok:
            if not (c in self.alpha or c in self.num):
                return False
        return True

    def isnum(self, tok):
        try:
            float(tok)
            return True
        except:
            return False

    def isop(self, tok):
        return tok in self.oplist

    def isass(self, tok):
        return tok == '='

    def islist(self, tok):
        return type(tok) == list

    def makeargs(self, *args):
        return tuple(sorted(set(args)))

    def tokenize(self, input):
        current = ''
        mode = ''
        newmode = ''
        result = [[]]
        for c in input:
            if (c in self.num and mode != 'var') or (c in ['e', 'E'] and mode == 'num') or (c == '-' and mode == 'num' and current[-1] in ['e', 'E']):
                newmode = 'num'
            elif c in self.alpha or (c in self.num and mode == 'var'):
                if mode == 'num':
                    mode == 'break'
                newmode = 'var'
            elif c in self.ops:
                if c == '-' and (self.isop(current) or (current == '' and (result[-1] == [] or self.isop(result[-1][-1])))):
                    newmode = 'adapt'
                else:
                    newmode = 'ops'
            elif c in self.seps:
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
                print('Unhandled character: %s' % c)
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
        opdict = {op: [] for op in self.oplist}

        # Classification of single tokens
        offset = 0
        for i in range(len(tlist)):
            tok = tlist[i]
            if self.isnum(tok):
                parsed.append(Number(tok))
            elif self.isvar(tok):
                parsed.append(Variable(tok))
            elif self.islist(tok):
                if i > 0 and isinstance(parsed[-1], Variable):
                    parsed[-1] = Function(str(parsed[-1]), self.makearguments(tok))
                    offset += 1
                else:
                    parsed.append(self.classify(tok))
            elif self.isop(tok):
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
        for i in sorted(opdict['**']+opdict['^']):
            try:
                i -= offset
                if i <= 0 or self.isop(parsed[i-1]) or self.isop(parsed[i+1]):
                    raise InvalidFormatError(self.deepjoin(tlist), "Exponentiation is a binary operator, requires one argument to the left and one to the right.")
                parsed[i-1:i+2] = [Exponent(parsed[i-1], parsed[i+1])]
                offset += 2
                for key in opdict:
                    opdict[key] = [n-2 if n > i else n for n in opdict[key]]
            except IndexError:
                raise InvalidFormatError(self.deepjoin(tlist), "Exponentiation is a binary operator, requires one argument to the left and one to the right.")
        offset = 0
        for i in sorted(opdict['*']+opdict['/']):
            try:
                i -= offset
                if i <= 0 or self.isop(parsed[i-1]) or self.isop(parsed[i+1]):
                    if i in opdict['*']:
                        raise InvalidFormatError(self.deepjoin(tlist), "Multiplication is a binary operator and requires one argument to the left and one to the right.")
                    elif i in opdict['/']:
                        raise InvalidFormatError(self.deepjoin(tlist), "Division is a binary operator and requires one argument to the left and one to the right.")
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
                    opdict[key] = [n-2 if n > i else n for n in opdict[key]]
            except IndexError:
                if i in opdict['*']:
                    raise InvalidFormatError(self.deepjoin(tlist), "Multiplication is a binary operator and requires one argument to the left and one to the right.")
                elif i in opdict['/']:
                    raise InvalidFormatError(self.deepjoin(tlist), "Division is a binary operator and requires one argument to the left and one to the right.")
        offset = 0
        for i in sorted(opdict['+']+opdict['-']):
            try:
                i -= offset
                if i <= 0 or self.isop(parsed[i-1]) or self.isop(parsed[i+1]):
                    if i in opdict['+']:
                        raise InvalidFormatError(self.deepjoin(tlist), "Addition is a binary operator and requires one argument to the left and one to the right.")
                    elif i in opdict['-']:
                        raise InvalidFormatError(self.deepjoin(tlist), "Subtraction is a binary operator and requires one argument to the left and one to the right.")
                if i in opdict['+']:
                    if isinstance(parsed[i-1], Sum):
                        parsed[i-1].summands.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Sum([parsed[i-1], parsed[i+1]])]
                elif i in opdict['-']:
                    if isinstance(parsed[i-1], Difference):
                        parsed[i-1].minuends.append(parsed[i+1])
                        parsed[i-1:i+2] = [parsed[i-1]]
                    else:
                        parsed[i-1:i+2] = [Difference(parsed[i-1], [parsed[i+1]])]
                offset += 2
                for key in opdict:
                    opdict[key] = [n-2 if n > i else n for n in opdict[key]]
            except IndexError:
                if i in opdict['+']:
                    raise InvalidFormatError(self.deepjoin(tlist), "Addition is a binary operator and requires one argument to the left and one to the right.")
                elif i in opdict['-']:
                    raise InvalidFormatError(self.deepjoin(tlist), "Subtraction is a binary operator and requires one argument to the left and one to the right.")
        try:
            return parsed[0]
        except IndexError:
            if len(tlist) == 0:
                raise InvalidFormatError(self.deepjoin(tlist), "Empty input list.")
            raise InvalidFormatError(self.deepjoin(tlist), "Unknown format error.")

    def makearguments(self, l):
        """Sub-function to create a list of arguments, to not be confused with a parenthesized expression."""
        return [self.classify(arg) for arg in self.split(l, ',')]

    def realize(self, tlist):
        """Takes a tokenized list of terms and constructs an expression tree from it."""
        if '=' in tlist:
            if self.isvar(tlist[0]) and self.isass(tlist[1]) and len(tlist) > 2 and '=' not in tlist[2:]:
                return VariableAssignment(tlist[0], self.classify(tlist[2:]))
            elif self.isvar(tlist[0]) and self.islist(tlist[1]) and self.isass(tlist[2]) and len(tlist) > 3 and '=' not in tlist[3:]:
                return FunctionAssignment(tlist[0], self.makearguments(tlist[1]), self.classify(tlist[3:]))
            else:
                reason = "Unkown format error."
                if tlist.count('=') > 1:
                    reason = "Nested or multiple assignments are not allowed."
                if not self.isvar(tlist[0]):
                    reason = "Only variables and functions can be assigned to."
                if tlist[-1] == '=':
                    reason = "Empty assignments are not allowed."
                raise InvalidFormatError(self.deepjoin(tlist), reason)
        return self.classify(tlist)

    def unapply(self, *args, **kwargs):
        """Turns all variables into functions over args, if any args appear within the variable."""
        exprs = deepcopy(self.exprlist)
        funcdict = {}
        for i in range(len(exprs)):
            expr = exprs[i]
            if isinstance(expr, (VariableAssignment, FunctionAssignment)):
                for func in funcdict:
                    expr.unapply(func, *funcdict[func])
                arglist = self.makeargs(*filter(lambda el: expr.contains(el), args))
                if len(arglist) > 0:
                    expr = FunctionAssignment(expr.name, arglist, expr.term)
                    exprs[i] = expr
                else:
                    expr = VariableAssignment(expr.name, expr.term)
                funcdict[expr.name] = arglist
        return exprs

    def unapplyall(self, *args):
        """Unapplies all parsed terms w.r.t. all combinations of arguments provided in *args. Results are stored for further application."""
        combinations = set([])
        for arg1 in args:
            for arg2 in args:
                combinations.add(self.makeargs(arg1, arg2, 'x', 'y'))
        for arglist in combinations:
            self.funclists[arglist] = self.unapply(*arglist)

    def integrate(self, expr, subs, area, args=('x', 'y')):
        """Numerical integration of expr, w.r.t. args and at the points provided in subs."""
        subs = [map(self.classify, [[term] for term in sub]) for sub in subs]
        return Product([Number(area), Sum([Function('abs', [deepcopy(expr).apply(args, sub) for sub in subs])])])

    def formatinput(self, input):
        """Formats Maple output to a list of separate expressions"""
        lines = input.strip().splitlines()
        terms = []
        term = ''
        for i in range(len(lines)):
            line = lines[i].strip()
            if line == '' or line.startswith('>'):
                continue
            if line.startswith('"['):
                line = line[2:]
            if line.endswith(']"'):
                line = line[:-2]
            elif line.endswith('\\'):
                line = line[:-1]
            elif i < len(lines)-1:
                line = line+", "
            line = line.replace('[', '_').replace(']', '')
            nestlevel = 0
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
            if nestlevel > 0:
                raise InvalidFormatError(input, "Parenthesis mismatch.")
            terms.append(term)
        return terms

    def parseterm(self, term):
        """Parses a single term into an internal expression"""
        return self.realize(self.tokenize(term))

    def parselist(self, l):
        """Takes a list of terms and parses them."""
        return [self.parseterm(el) for el in l]

    def parse(self, input):
        """Parse a comma-separated list of terms."""
        self.exprlist = self.parselist(self.formatinput(input))

    def fullparse(self, input, *args):
        """Parses a comma-seperated list of terms, stores the results and then unapplies all combinations of the provided arguments and stores the results separately."""
        self.parse(input)
        self.unapplyall(*args)

    def setvars(self, exprs, name, val):
        """Set a variable within the given expression list to a certain value."""
        for expr in exprs:
            expr.setvar(name, val)

    def setvar(self, name, val):
        self.setvars(self.exprlist, name, val)
        for args in self.funclists:
            self.setvars(self.funclists[args], name, val)

    def simplify(self, exprlist):
        """Takes a list of expressions and simplifies them much as possible with considerations to set variables. Returns a (possibly shortened) list of expressions."""
        exprs = deepcopy(exprlist)
        valdict = {}
        for i in range(len(exprs)):
            expr = exprs[i]
            try:
                for name, value in valdict:
                    expr.setvar(name, value)
                valdict[expr.name] = float(str(expr.term))
                exprs[i:i+1] = []
            except ValueError:
                continue

    def gnuformat(self, *args, **kwargs):
        """Format the current expression tree to a Gnuplot string.
            args contains the variables which it should plot.
            kwargs can contain:
                integrate: Domain type to integrate
                    Line
                    RTriangle
                start: Start of x integration domain
                end: End of x integration domain
                stops: Number of partitions of the x domain
            Higher dimensional domains will be integrated evenly."""
        args = self.makeargs(*args)
        if len(args) < 1 or len(args) > 2:
            raise InvalidCallError('('+', '.join(args)+')', 'Invalid number of arguments. One argument for a 2D plot, two for a 3D plot.')
        try:
            exprs = self.funclists[str(args)]
        except KeyError:
            self.funclists[str(args)] = self.unapply(*args+('x', 'y'))
            exprs = self.funclists[str(args)]
        if 'integrate' in kwargs:
            start = kwargs['start'] if 'start' in kwargs else 0
            end = kwargs['end'] if 'end' in kwargs else 1
            stops = kwargs['stops'] if 'stops' in kwargs else 5
            length = math.fabs(end-start)/stops
            subs = None
            if kwargs['integrate'] == 'RTriangle':
                combinations = []
                for i in range(stops):
                    for j in range(stops-i):
                        combinations.append((i,j))
                subs = map(lambda (x, y): (x+1.0/6.0, y+1.0/6.0), combinations)+map(lambda (x, y): (x+5.0/6.0, y+5.0/6.0), combinations)
                subs = map(lambda (x, y): (start+x*length, start+y*length), subs)
                subs = filter(lambda (x, y): x+y <= end, subs)
                area = length**2/2.0
            elif kwargs['integrate'] == 'Line':
                subs = [(x+0.5)/length for x in range(stops)]
                area = length
            if not subs or not area:
                if isinstance(kwargs['integrate'], str):
                    raise InvalidOperationError('Integration domain %s unknown or not supported.' % kwargs['integrate'])
                else:
                    raise InvalidOperationError('No integration domain specified.')
            area /= len(subs)
            if isinstance(exprs[-1], (VariableAssignment, FunctionAssignment)):
                exprs[-1].term = self.integrate(exprs[-1].term, subs, area)
                if isinstance(exprs[-1], FunctionAssignment):
                    exprs[-1].args = filter(lambda arg: arg not in ['x', 'y'], exprs[-1].args)
            else:
                exprs[-1] = self.integrate(exprs[-1], subs, area)
        if len(args) == 1:
            argformat = 'set dummy %s' % args
        elif len(args) == 2:
            argformat = 'set dummy %s, %s' % args
        exprstr = '\n'.join([str(expr) for expr in exprs])
        plotcmd = ('s' if len(args) == 2 else '')+'plot %s'
        if isinstance(exprs[-1], FunctionAssignment):
            plotstr = plotcmd % exprs[-1].name+'('+', '.join(args)+')'
        elif isinstance(exprs[-1], VariableAssignment):
            plotstr = plotcmd % exprs[-1].name
        else:
            plotstr = plotcmd % exprs[-1]
        return '%s\n%s\n%s' % (argformat, exprstr, plotstr)

if __name__ == '__main__':
    #termstring = 't1 = 0.5*3*h[1]+u[1]*h[2], t2 = 2*t1 + 3/(h[1]*h[2]*u[2]), t3 = 2h[1]+4t2 u[1], t4 = 2 h[1]**2 + t2^3 + t3*u[2]+ 5 t1 *(h[1]*h[2]), t15= t4*u[1]*u[2]*h[1]^2*h[2]**2 +  h[1], t1642 = 0.12t15+h[1]'
    termstring = 't1 = 0.5*3*h[1]+x*u[1]*h[2], t2 = y+2*t1 + 3/(h[1]*h[2]*u[2]), t3(h[2]) = t2 + h[2]*x + u[2]*y'
    print('Not supposed to be run on its own. Demonstrative run using the following input:\n%s\n' % '\n'.join(termstring.split(', ')))

    p = Mathparser()
    p.parse(termstring)

    print('Parsed expressions:')
    print('\n'.join(str(expr) for expr in p.exprlist)+'\n')
    p.unapplyall('h_1', 'h_2', 'u_1', 'u_2')
    print(p.gnuformat('h_1', 'h_2', integrate='RTriangle'))

    #print('\nInternal representation:')
    #print('\n'.join([repr(expr) for expr in p.exprlist]))
    #print('\nGnuplot formatted string:')
    #print(p.gnuformat('h_1', 'u_1'))
