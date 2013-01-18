class InvalidFormatException(Exception):
    def __init__(self, line, reason):
        self.line = line
        self.reason = reason
    def __str__(self):
        return repr(self.line)+': '+repr(self.reason)
    def __repr__(self):
        return repr(self.line)+': '+repr(self.reason)

class Term():
    def apply(self, name, val):
        return None

class Number(Term):
    def __init__(self, number):
        self.number = float(number)
    def __repr__(self):
        return 'Number('+repr(self.number)+')'
    def __str__(self):
        return str(self.number)

class Sum(Term):
    def __init__(self, summands):
        self.summands = summands
    def __repr__(self):
        return 'Addition('+', '.join([repr(summand) for summand in self.summands])+')'
    def __str__(self):
        summands = []
        acc = 0
        for summand in [str(summand) for summand in self.summands]:
            try:
                acc += float(summand)
            except ValueError:
                summands.append(summand)
        if acc == 0:
            return '+'.join(summands)
        if len(summands) == 0:
            return str(acc)
        return str(acc)+'+'+'+'.join(summands)
    def apply(self, name, val):
        for summand in self.summands:
            summand.apply(name, val)

class Difference(Term):
    def __init__(self, subtrahend, minuends):
        self.subtrahend = subtrahend
        self.minuends = minuends
    def __repr__(self):
        return 'Difference('+repr(self.subtrahend)+', '+', '.join([repr(minuend) for minuend in self.minuends])+')'
    def __str__(self):
        minuends = []
        acc = 0
        for minuend in [str(minuend) for minuend in self.minuends]:
            try:
                acc += float(minuend)
            except ValueError:
                minuends.append(minuend)
        if acc == 0:
            return str(self.subtrahend)+'-'+'-'.join(minuends)
        try:
            subtrahend = str(float(subtrahend) - acc)
        except ValueError:
            res.append(str(acc))
        if len(minuends) == 0:
            return subtrahend
        return subtrahend+'-'.join(minuends)
    def apply(self, name, val):
        self.subtrahend.apply(name, val)
        for minuend in self.minuends:
            minuend.apply(name, val)

class Product(Term):
    def __init__(self, factors):
        self.factors = factors
    def __repr__(self):
        return 'Product('+', '.join([repr(factor) for factor in self.factors])+')'
    def __str__(self):
        factors = [] #['('+str(factor)+')' if isinstance(factor, (Sum, Difference)) else str(factor) for factor in self.factors]
        acc = 1
        #print(self.factors)
        #print([str(f) for f in self.factors])
        for factor in [str(factor) for factor in self.factors]:
            try:
                acc *= float(factor)
            except ValueError:
                factors.append(factor)
        if acc == 0:
            return '*'.join(factors)
        if len(factors) == 0:
            return str(acc)
        return str(acc)+'*'+'*'.join(factors)
    def apply(self, name, val):
        for factor in self.factors:
            factor.apply(name, val)

class Quotient(Term):
    def __init__(self, dividend, divisors):
        self.dividend = dividend
        self.divisors = divisors
    def __repr__(self):
        return 'Division('+repr(self.dividend)+', '+', '.join([repr(divisor) for divisor in self.divisors])+')'
    def __str__(self):
        dividend = str('('+str(self.dividend)+')' if isinstance(self.dividend, (Sum, Difference)) else str(self.dividend))
        divisors = []
        acc = 1
        for divisor in ['('+str(divisor)+')' if isinstance(divisor, (Sum, Difference)) else str(divisor) for divisor in self.divisors]:
            try:
                acc *= float(divisor)
            except ValueError:
                divisors.append(divisor)
        if acc == 0:
            return str(self.dividend)+'/'+'/'.join(divisors)
        try:
            dividend = str(float(dividend) / acc)
        except ValueError:
            res.append(str(acc))
        if len(divisors) == 0:
            return dividend
        return dividend+'/'.join(divisors)
    def apply(self, name, val):
        self.dividend.apply(name, val)
        for divisor in self.divisors:
            divisor.apply(name, val)

class Exponent(Term):
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp
    def __repr__(self):
        return 'Exponent('+repr(self.base)+', '+repr(self.exp)+')'
    def __str__(self):
        base = str('('+str(self.base)+')' if isinstance(self.base, (Sum, Difference, Product, Quotient)) else str(self.base))
        exp = str('('+str(self.exp)+')' if isinstance(self.exp, (Sum, Difference, Product, Quotient)) else str(self.exp))
        try:
            return str(float(base)**float(exp))
        except ValueError:
            return base+'**'+exp
    def apply(self, name, val):
        self.base.apply(name, val)
        self.exp.apply(name, val)

class VariableAssignment(Term):
    def __init__(self, name, term):
        self.name = name
        self.term = term
    def __repr__(self):
        return 'VariableAssignment('+repr(self.name)+', '+repr(self.term)+')'
    def __str__(self):
        return str(self.name)+' = '+str(self.term)
    def apply(self, name, val):
        self.term.apply(name, val)

class FunctionAssignment(Term):
    def __init__(self, name, args, term):
        self.name = name
        self.args = args
        self.term = term
    def __repr__(self):
        return 'FunctionAssignment('+repr(self.name)+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), '+repr(self.term)+')'
    def __str__(self):
        return str(self.name)+'('+', '.join([str(arg) for arg in self.args])+') = '+str(self.term)
    def apply(self, name, val):
        self.term.apply(name, val)

class Variable(Term):
    def __init__(self, name):
        self.name = name
        self.val = None
    def __repr__(self):
        if self.val != None:
            return 'Variable('+repr(self.name)+', '+repr(self.val)+')'
        else:
            return 'Variable('+repr(self.name)+', Unassigned)'
    def __str__(self):
        if self.val != None:
            return str(self.val)
        else:
            return str(self.name)
    def apply(self, name, val):
        if self.name == name:
            self.val = val

class Function(Term):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.val = None
    def __repr__(self):
        if self.val != None:
            return 'Function('+repr(self.name)+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), '+repr(self.val)+')'
        else:
            return 'Function('+repr(self.name)+', Arguments('+', '.join([repr(arg) for arg in self.args])+'), Unassigned)'
    def __str__(self):
        if self.val != None:
            return str(self.name)+'('+', '.join([str(arg) for arg in self.args])+')'
            return str(self.val)
        else:
            return str(self.name)+'('+', '.join([str(arg) for arg in self.args])+')'
    def apply(self, name, val):
        if self.name == name:
            self.val = val

class Mathparser():
    def __init__(self):
        self.num = '0123456789.'
        self.alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        self.ops = '+-*/^'
        self.seps = '=,'
        self.oplist = ['+', '-', '*', '/', '^', '**']

        self.exprlist = []

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

    def tokenize(self, input):
        current = ''
        mode = ''
        newmode = ''
        result = [[]]
        for c in input:
            if (c in self.num and mode != 'var') or (c in ['e', 'E'] and mode == 'num'):
                newmode = 'num'
            elif c in self.alpha or (c in self.num and mode == 'var'):
                if mode == 'num':
                    mode == 'break'
                newmode = 'var'
            elif c in self.ops:
                if c == '-' and (result[-1] == [] or self.isop(result[-1][-1])):
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

    def realize(self, tlist):
        def makearguments(l):
            return [classify(arg) for arg in self.split(l, ',')]
    
        def classify(tlist):
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
                        parsed[-1] = Function(str(parsed[-1]), makearguments(tok))
                        offset += 1
                    else:
                        parsed.append(classify(tok))
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
                        raise InvalidFormatException(self.deepjoin(tlist), 'Exponentiation is a binary operator, requires one argument to the left and one to the right.')
                    parsed[i-1:i+2] = [Exponent(parsed[i-1], parsed[i+1])]
                    offset += 2
                    for key in opdict:
                        opdict[key] = [n-2 if n > i else n for n in opdict[key]]
                except IndexError:
                    raise InvalidFormatException(self.deepjoin(tlist), 'Exponentiation is a binary operator, requires one argument to the left and one to the right.')
            offset = 0
            for i in sorted(opdict['*']+opdict['/']):
                try:
                    i -= offset
                    if i <= 0 or self.isop(parsed[i-1]) or self.isop(parsed[i+1]):
                        if i in opdict['*']:
                            raise InvalidFormatException(self.deepjoin(tlist), 'Multiplication is a binary operator and requires one argument to the left and one to the right.')
                        elif i in opdict['/']:
                            raise InvalidFormatException(self.deepjoin(tlist), 'Division is a binary operator and requires one argument to the left and one to the right.')
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
                            parsed[i-1:i+2] = [Quotient(parsed[i-1], parsed[i+1])]
                    offset += 2
                    for key in opdict:
                        opdict[key] = [n-2 if n > i else n for n in opdict[key]]
                except IndexError:
                    if i in opdict['*']:
                        raise InvalidFormatException(self.deepjoin(tlist), 'Multiplication is a binary operator and requires one argument to the left and one to the right.')
                    elif i in opdict['/']:
                        raise InvalidFormatException(self.deepjoin(tlist), 'Division is a binary operator and requires one argument to the left and one to the right.')
            offset = 0
            for i in sorted(opdict['+']+opdict['-']):
                try:
                    i -= offset
                    if i <= 0 or self.isop(parsed[i-1]) or self.isop(parsed[i+1]):
                        if i in opdict['+']:
                            raise InvalidFormatException(self.deepjoin(tlist), 'Addition is a binary operator and requires one argument to the left and one to the right.')
                        elif i in opdict['-']:
                            raise InvalidFormatException(self.deepjoin(tlist), 'Subtraction is a binary operator and requires one argument to the left and one to the right.')
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
                            parsed[i-1:i+2] = [Difference(parsed[i-1], parsed[i+1])]
                    offset += 2
                    for key in opdict:
                        opdict[key] = [n-2 if n > i else n for n in opdict[key]]
                except IndexError:
                    if i in opdict['+']:
                        raise InvalidFormatException(self.deepjoin(tlist), 'Addition is a binary operator and requires one argument to the left and one to the right.')
                    elif i in opdict['-']:
                        raise InvalidFormatException(self.deepjoin(tlist), 'Subtraction is a binary operator and requires one argument to the left and one to the right.')
        
            return parsed[0]

        if '=' in tlist:
            if self.isvar(tlist[0]) and self.isass(tlist[1]) and len(tlist) > 2 and '=' not in tlist[2:]:
                return VariableAssignment(tlist[0], classify(tlist[2:]))
            elif self.isvar(tlist[0]) and self.islist(tlist[1]) and self.isass(tlist[2]) and len(tlist) > 3 and '=' not in tlist[3:]:
                return FunctionAssignment(tlist[0], tlist[1], classify(tlist[3:]))
            else:
                reason = "Unkown format error."
                if tlist.count('=') > 1:
                    reason = "Nested or multiple assignments are not allowed."
                if not isvar(tlist[0]):
                    reason = "Only variables and functions can be assigned to."
                if tlist[len(tlist)-1] == '=':
                    reason = "Empty assignments are not allowed."
                raise InvalidFormatException(self.deepjoin(tlist), reason)
        return classify(tlist)

    # Parses a single term
    def parseline(self, term):
        return self.realize(self.tokenize(term))

    # Formats Maple output to list of terms
    def formatinput(self, input):
        lines = []
        for line in input.splitlines():
            line = line.strip()
            if line == '' or line.startswith('>'):
                continue
            if line.startswith('"['):
                line = line[2:]
            if line.endswith(']"'):
                line = line[:-2]
            elif line.endswith('\\'):
                line = line[:-1]
            else:
                line = line+", "
            lines.append(line)
        return ''.join(lines).split(', ')

    # Takes a list of terms and parses them
    def parselist(self, l):
        return [self.parseline(el) for el in l]

    def parse(self, input):
        self.exprlist = self.parselist(self.formatinput(input))

    def setvar(self, name, val):
        for expr in self.exprlist:
            expr.apply(name, val)

    def gnuformat(self):
        return '\n'.join([str(expr) for expr in self.exprlist])

if __name__ == '__main__':
    termstring = '"[13+2 e5, 1+2+3, 1+2*3, (1+2)(3), t2 = 2*t1 + 3, a(x) = 2+4t2, t4 = 2 a(x)**2 + 5 t1 (y)]"'
    print('Not supposed to be run on its own. Demonstrative run using the following input:\n%s' % termstring)

    p = Mathparser()

    p.parse(termstring)

    print('\nGnuplot formatted string:')
    print(p.gnuformat())
    print('\nInternal representation:')
    print('\n'.join([repr(expr) for expr in p.exprlist]))

    print('\nSetting variable t1 to 5')

    p.setvar('t1', 5)

    print('\nGnuplot formatted string:')
    print(p.gnuformat())
    print('\nInternal representation:')
    print('\n'.join([repr(expr) for expr in p.exprlist]))
