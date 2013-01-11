import re

class Term(object):
    def __str__(self):
        return "[Some term]"
    def simplify(self):
        return self
    def substitute(self, s, d):
        """
        Substitutes all variables properly.
        * s: List of strings (representing variable names)
        * d: List of terms to replace the variables by
        """
        assert type(s)==type(d)
        if type(s) == str:
            source = [s]
            dest = [d]
        else:
            source = s
            dest = d
        assert len(source)==len(dest)
        return self.substitute_intern(source, dest)
    def substitute_intern(self, source, dest):
        raise Exception("To be implemented in subclass!")

class Variable(Term):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def substitute_intern(self, source, dest):
        for (s, d) in zip(source, dest):
            if self.name == s:
                return d
        return self

class Number(Term):
    def __init__(self, number):
        self.number = number
    def __str__(self):
        return str(self.number)
    def substitute_intern(self, s, d):
        return self

class Sum(Term):
    def __init__(self, *args):
        self.summands = args
    def __str__(self):
        return "+".join(["(%s)"%str(x) for x in self.summands])
    def simplify(self):
        if len(self.summands) == 1:
            return self.summands[0].simplify()
        return self
    def substitute_intern(self, source, dest):
        return Sum(*[x.substitute_intern(source, dest) for x in self.summands])

class Product(Term):
    def __init__(self, *args):
        self.factors = args
    def __str__(self):
        return "*".join(["(%s)"%str(x) for x in self.factors])
    def substitute_intern(self, source, dest):
        return Product(*[x.substitute_intern(source, dest) for x in self.factors])

class Minus(Term):
    def __init__(self, term):
        self.term = term
    def __str__(self):
        return "-%s"%(str(self.term))
    def substitute_intern(self, source, dest):
        return Minus(self.term.substitute_intern(source, dest))

class Quotient(Term):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
    def __str__(self):
        return "(%s)/(%s)"%(str(self.numerator), str(self.denominator))
    def substitute_intern(self, source, dest):
        return Quotient(self.numerator.substitute_intern(source, dest),
                        self.denominator.substitute_intern(source, dest))

class Power(Term):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __str__(self):
        return "(%s)**(%s)"%(str(self.base), str(self.exponent))
    def substitute_intern(self, source, dest):
        return Power(self.base.substitute_intern(source, dest),
                     self.exponent.substitute_intern(source, dest))

class Funcall(Term):
    def __init__(self, fun, params):
        self.fun = fun
        self.params = params
    def expand(self):
        return self.fun.eval(self.params)
    def substitute_intern(self, source, dest):
        return Function(self.fun, [x.substitute_intern(source, dest) for x in self.params])
    def __str__(self):
        return str(self.fun.eval(self.params))

class Function(object):
    def __init__(self, name, arguments, term):
        self.name = name
        self.arguments = arguments
        self.term = term
    def eval(self, params):
        assert len(params)==len(self.arguments)
        return self.term.substitute(self.arguments, params)
    def __str__(self):
        return "%s(%s)"%(self.name, ", ".join(self.arguments))

def split_string_parenthesized(s, split):
    count = 0
    splits = [-1]
    for (i, c) in enumerate(s):
        if c=="(":
            count = count + 1
        if c==")":
            count = count - 1
        if c==split and count==0:
            splits.append(i)
    splits.append(len(s))
    parts = [s[splits[i]+1:splits[i+1]] for i in xrange(len(splits)-1)]
    return parts

def make_string_better_parsable(s):
    s = re.sub(r"\++", "+", s)
    s = re.sub(r"\+-", "-", s)
    s = s.replace("-", "+-")
    if s[0]=="+":
        s = s[1:]
    return s

def split_sum_and_prod(s, fun_environment={}):
    s = make_string_better_parsable(s)
    # first, we try for a sum
    summand_str = split_string_parenthesized(s, "+")
    if len(summand_str) > 1:
        print summand_str
        return Sum(*[parse_string(x) for x in summand_str])
    # if we come here, we have no sum, so we try product
    print "No sum found."
    factors_str = split_string_parenthesized(s, "*")
    if len(factors_str) > 1:
        print factors_str
        return Product(*[parse_string(x) for x in factors_str])
    # if we come here, we have no sum or product so we try exponentiation
    print "No product found."
    power_str = split_string_parenthesized(s, "^")
    if len(power_str)==2:
        print power_str 
        return Power(parse_string(power_str[0]), parse_string(power_str[1]))
    # if we come here, we have no sum, product or exponent, thus
    # we try for a quotient
    quotient_str = split_string_parenthesized(s, "/")
    if len(quotient_str)>1:
        print quotient_str
        def quotientizer(a,b):
            return Quotient(a, parse_string(b))
        return reduce(quotientizer, quotient_str[1:], parse_string(quotient_str[0]))
    # none of the above? Probably a function call
    print "No sum, product or exponentiation"
    if re.match(r"^[a-zA-Z_]+(\d*[a-zA-Z_]*)*\(.*\)$", s) is not None:
        print "function call!"
        fname, fparams = s.split("(", 1)
        fparams = fparams[:-1]
        print fname, fparams
        count = 0
        splits = [-1]
        for (i, c) in enumerate(fparams):
            if c=="(":
                count = count + 1
            if c==")":
                count = count - 1
            if c=="," and count==0:
                splits.append(i)
        splits.append(len(s))
        params_str = [fparams[splits[i]+1:splits[i+1]] for i in xrange(len(splits)-1)]
        print params_str
        print fun_environment
        return Funcall(fun_environment[fname], [parse_string(x) for x in params_str])

def parse_string(s, fun_environment={}):
    # raw_input ("Parsing %s"%s)
    # trim outer brackets
    if s[0]=="(" and s[-1]==")":
        return parse_string(s[1:-1])
    # is string "negated"?
    if s[0]=="-":
        return Minus(parse_string(s[1:]))
    # is it a simple number?
    if re.search(r"^-?\d+(?:\.d+)?(?:[eE]d+)?$", s) is not None:
        return Number(float(s))
    # is it a simple variable?
    if re.search(r"^([a-zA-Z_]\d*[a-zA-Z_]*)$", s) is not None:
        return Variable(s)
    # probably something else
    return split_sum_and_prod(s, fun_environment)

f = Function("f", ["a","b","c"], parse_string("a+b+c"))
x = parse_string("x+y")
result = parse_string("f(a,b,(1+c))", {"f": f})
print result

print parse_string("2^((a-c)^3)")
# print parse_string("a*(2*b-c)*((2+b*2)-2)")
print parse_string("a/b/c/d")
